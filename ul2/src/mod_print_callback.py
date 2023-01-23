import torch
import sys

import transformers
from transformers import PreTrainedModel

from composer import Callback, Logger, State
from composer.utils import dist
from src.data_denoising import MixtureOfDenoisersCollator
from src import utils

from typing import Any, Dict, List, Optional

__all__ = ['MixtureOfDenoisersPrinterCallback']

class MixtureOfDenoisersPrinterCallback(Callback):

    def __init__(
        self,
        tokenizer_name: str='t5-base',
        print_frequency: int=500,
        max_length: int=128,
        raise_on_failure: bool=False,
        generate_kwargs: Optional[Dict[str, Any]]=None,
    ) -> None:

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        utils.adapt_tokenizer_for_denoising(self.tokenizer, num_sentinel_tokens=100)

        self.print_frequency = int(print_frequency)
        assert self.print_frequency >= 1
        self._steps_since_last_print = print_frequency # This will cause it to print on the first batch, will be reset to 0 each print.

        self.max_length = int(max_length)
        assert self.max_length >= 1

        self._raise_on_error = bool(raise_on_failure)

        self._generate_kwargs = generate_kwargs if generate_kwargs else {}

        self._verified_collator = False
        self._decoder_only_format = None

    def init(self, state: State, logger: Logger) -> None:
        """Verifies model class."""
        del logger # unused

        assert hasattr(state.model, 'model'), "This callback requires that state.model is a HuggingFaceWrapper instance."
        hf_model = state.model.model
        assert isinstance(hf_model, PreTrainedModel), "This class only works for models that subclass `transformers.PreTrainedModel`."

    def after_dataloader(self, state: State, logger: Logger) -> None:
        """Periodically prints formatted outputs generated by the model"""
        del logger # unused

        if dist.get_global_rank() != 0:
            return

        if not self._verified_collator:
            assert isinstance(state.train_dataloader.collate_fn, MixtureOfDenoisersCollator), "The dataloader must use the `MixtureOfDenoisersCollator` as the collate function."
            self._verified_collator = True
            self._decoder_only_format = bool(state.train_dataloader.collate_fn.decoder_only_format)

        self._steps_since_last_print += 1
        if self._steps_since_last_print < self.print_frequency:
            return

        if state.is_model_ddp:
            model = state.model.module.model # Extract from DDP wrapping
        else:
            model = state.model.model

        input_str, target_str, output_str = self._generate_input_target_output_strings(state.batch, model)
        
        try:
            example_string = self._format_strings(input_str, target_str, output_str)
            print(f'\n\n{self.__class__.__name__} printout...\n{example_string}\n', file=sys.stdout, flush=True)
            
        except AssertionError:
            if self._raise_on_error:
                raise
            print(
                f'\n\n{self.__class__.__name__}: Failed to parse input/target/output strings. This is expected if the model is early in training. Set `raise_on_failure=True` if you want this to failure generate an error.\n',
                file=sys.stdout, flush=True
            )

        self._steps_since_last_print = 0

    def _generate_input_target_output_strings(self, batch: Dict[str, torch.TensorType], model: PreTrainedModel) -> List[str]:
        """Generates outputs from the model and decodes the input, target, and output tokens into strings."""
        # Take inputs from the first batch element, and remove padding
        if self._decoder_only_format:
            non_padding = batch['attention_mask'][0].to(torch.bool)
            prefix_mask = batch['bidirectional_mask'][0, non_padding].to(torch.bool)
            target_mask = torch.logical_not(prefix_mask)
            prefix_and_target = batch['input_ids'][:1, non_padding]
            input_ids = prefix_and_target[:, prefix_mask] # This is the prefix
            labels = prefix_and_target[:, target_mask] # This is the target

            # Generate the output prediction up to the maximum sequence length
            outputs = model.generate(input_ids,
                                     max_new_tokens=self.max_length,
                                     pad_token_id=self.tokenizer.pad_token_id,
                                     **self._generate_kwargs)

            # Strip the prefix from the output
            outputs = outputs[:, prefix_mask.sum():]

        else:
            attn_inputs = batch['attention_mask'][0].to(torch.bool)
            attn_labels = batch['decoder_attention_mask'][0].to(torch.bool)
            input_ids = batch['input_ids'][:1, attn_inputs]
            labels = batch['labels'][:1, attn_labels]
        
            # Generate the output prediction up to the maximum sequence length
            outputs = model.generate(input_ids, max_length=self.max_length, **self._generate_kwargs)
        
        # Decode the token sequences into strings
        input_str = self.tokenizer.decode(input_ids[0])
        target_str = self.tokenizer.decode(labels[0])
        output_str = self.tokenizer.decode(outputs[0])
        
        return input_str, target_str, output_str

    @staticmethod
    def _format_strings(input_str: str, target_str: str, output_str: str) -> None:
        def split_input(s: str) -> List[str]:
            sentinels = [f'<extra_id_{i}>' for i in range(100)]
            if sentinels[0] not in s:
                s = s + sentinels[0]
            outs = []
            substr, s = s.split(sentinels[0])
            outs.append(substr)
            for curr_sent in sentinels[1:]:
                if curr_sent not in s:
                    outs.append(s)
                    break
                substr, s = s.split(curr_sent)
                outs.append(substr)
            return outs

        def split_output(s: str) -> List[str]:
            sentinels = [f'<extra_id_{i}>' for i in range(100)]
            if sentinels[0] not in s:
                s = sentinels[0] + s
            outs = [s.split(sentinels[0])[1]]
            for s_idx in range(1, 100):
                prev_s = outs.pop(-1)
                curr_sent = sentinels[s_idx]
                if curr_sent not in prev_s:
                    outs.append(prev_s)
                    break
                splits = prev_s.split(curr_sent)
                prev_s = splits[0]
                curr_s = curr_sent.join(splits[1:])
                outs.append(prev_s)
                outs.append(curr_s)
            return outs

        def build_context_mask_string(context_chunks: List[str], masked_chunks: List[str]) -> str:
            str_chunks = []
            for idx, (c_, m_) in enumerate(zip(context_chunks, masked_chunks)):
                str_chunks.append(c_)
                if idx == len(context_chunks) - 1:
                    break
                str_chunks.append(m_)
            return ' '.join(str_chunks)

        # Input string chunks are yellow
        i_split = ["\033[93m {}\033[00m" .format(s) for s in split_input(input_str)]
        # Target string chunks are red
        t_split = ["\033[91m {}\033[00m" .format(s) for s in split_output(target_str)]
        # Output string chunks are green
        o_split = ["\033[92m {}\033[00m" .format(s) for s in split_output(output_str)]

        actual = build_context_mask_string(i_split, t_split)
        predicted = build_context_mask_string(i_split, o_split)

        return f'Actual:\n{actual}\nPredicted:\n{predicted}'
        

        