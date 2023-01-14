# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""
Build a StreamingC4 dataset and dataloader for training.
"""

import os
import sys
from itertools import islice
from typing import Any, Dict, Iterator, List, Mapping, Optional, Union

import random
import torch
import numpy as np
import transformers
from transformers import T5Tokenizer, T5TokenizerFast
from omegaconf import OmegaConf as om
from streaming import Dataset
from torch.utils.data import DataLoader


class StreamingC4(Dataset):
    """
    Implementation of the C4 (Colossal Cleaned Common Crawl) dataset using mosaicml-streaming's Dataset V2.

    Args:
        remote (str): Remote directory (S3 or local filesystem) where dataset is stored.
        local (str): Local filesystem directory where dataset is cached during operation.
        split (str): The dataset split to use, either 'train' or 'val'.
        shuffle (bool): Whether to shuffle the samples in this dataset.
        prefetch (int): Target number of samples remaining to prefetch while iterating.
        tokenizer_name (str): The name of the HuggingFace tokenizer to use to tokenize samples.
        max_seq_len (int): The max sequence length of each token sample.
        group_method (str): How to group text samples into token samples. Supports 'concat' (default) or 'truncate'.
        retry (int): Number of download re-attempts before giving up. Default: 2.
        timeout (float): How long to wait for shard to download before raising an exception. Default: 120 sec.
        batch_size (Optional[int]): Hint batch_size that will be used on each device's DataLoader. Default: ``None``.
    """

    def __init__(self,
                 remote: str,
                 local: str,
                 split: str,
                 shuffle: bool,
                 prefetch: int,
                 tokenizer_name: str,
                 max_seq_len: int,
                 group_method: str = 'concat',
                 retry: int = 2,
                 timeout: float = 120,
                 batch_size: Optional[int] = None):
        # Validation
        if split not in ['train', 'val']:
            raise ValueError(f"split='{split}' must be one of ['train', 'val'].")
        if group_method not in ['truncate', 'concat']:
            raise ValueError(f"group_method='{group_method}' must be one of ['truncate', 'concat'].")

        # Build Dataset
        super().__init__(remote=remote,
                         local=local,
                         split=split,
                         shuffle=shuffle,
                         prefetch=prefetch,
                         keep_zip=False,
                         retry=retry,
                         timeout=timeout,
                         hash=None,
                         batch_size=batch_size)
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len
        self.group_method = group_method

        # Build tokenizer
        os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_name)
        if self.tokenizer.pad_token is None:
            # Some tokenizers (e.g. GPT2 tokenizer) have no padding token which causes bugs
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # suppress warnings when using group_method='concat' and no truncation
        self.tokenizer.model_max_length = int(1e30)

    # How to tokenize a text sample to a token sample
    def _tokenize(self, text_sample):
        if self.group_method == 'truncate':
            truncation = True
            padding = True
            max_length = self.max_seq_len
        elif self.group_method == 'concat':
            truncation = False
            padding = False
            max_length = None
        else:
            raise ValueError(f"Got unknown group_method='{self.group_method}'.")
        return self.tokenizer(text_sample['text'], truncation=truncation, padding=padding, max_length=max_length)

    # How to process a sample
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text_sample = super().__getitem__(idx)
        token_sample = self._tokenize(text_sample)
        return token_sample

    # Define iterable over samples
    # Usually this can be left alone and inherited directly from super() class StreamingDataset, but concatenating samples is custom behavior.
    # If group_method=='truncate', we simply return the token sample.
    # If group_method=='concat', then we keep fetching token samples until we fill up max_seq_len.
    def __iter__(self) -> Iterator[Any]:
        if self.group_method == 'truncate':
            iterator = super().__iter__()
            yield from iterator

        elif self.group_method == 'concat':
            buffer = {}
            while True:
                iterator = super().__iter__()
                for sample in iterator:

                    for k, v in sample.items():
                        buffer[k] = buffer.get(k, []) + v
                    while len(buffer['input_ids']) >= self.max_seq_len:
                        concat_sample = {}
                        for k, v in buffer.items():
                            concat_sample[k] = v[:self.max_seq_len]
                            buffer[k] = v[self.max_seq_len:]
                        yield concat_sample
        else:
            raise ValueError(f"Got unknown group_method='{self.group_method}'.")

    # Define length
    # Usually this can be left alone and inherited directly from super() class Dataset, but concatenating samples is custom behavior.
    # If group_method=='truncate', we simply return the # samples.
    # If group_method=='concat', we repeat forever, and we don't have a defined length.
    def __len__(self) -> int:
        if self.group_method == 'truncate':
            return super().__len__()
        elif self.group_method == 'concat':
            return None
        else:
            raise ValueError(f"Got unknown group_method='{self.group_method}'.")


class MixtureOfDenoisersCollator:
    def __init__(
        self,
        tokenizer: Union[T5Tokenizer, T5TokenizerFast],
        max_seq_length: int,
        decoder_only_format: bool = False,
        span_mean_lengths_and_ratios: Optional[Union[List[List[float]], List[float]]] = None,
        sequence_mask_ratios: Optional[Union[List[float], float]] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.decoder_only_format = decoder_only_format

        if not isinstance(self.tokenizer, (T5Tokenizer, T5TokenizerFast)):
            raise TypeError('Tokenizer must be from the T5 family.')

        self._denoiser_tags = [
            '[NLU]', # "Regular" span corruption
            '[NLG]', # "Extreme" span corruption
            '[S2S]', # Sequential denoising
        ]
        self._denoiser_tag_token_ids = {} # To be prepended to `input_ids` when corrupting
        for tag in self._denoiser_tags:
            tag_tokens = self.tokenizer(tag).input_ids
            special = self.tokenizer.get_special_tokens_mask(tag_tokens, already_has_special_tokens=True)
            self._denoiser_tag_token_ids[tag] = [tag_token for tag_token, spc in zip(tag_tokens, special) if not spc]

        self._noisers = []
        
        # Add "noisers" for the span corruption denoising task
        self.span_mean_lengths_and_ratios = span_mean_lengths_and_ratios
        if self.span_mean_lengths_and_ratios is None:
            self.span_mean_lengths_and_ratios = []
        elif isinstance(self.span_mean_lengths_and_ratios[0], (int, float)):
            assert len(self.span_mean_lengths_and_ratios) == 2, "`span_mean_lengths_and_ratios` must be a pair of [mean_length, mask_ratio] or a list of such pairs."
            self.span_mean_lengths_and_ratios = [self.span_mean_lengths_and_ratios]
        for span_mean_length, span_mask_ratio in self.span_mean_lengths_and_ratios:
            assert span_mean_length > 0, "All span mean lengths must be positive."
            assert 0 < span_mask_ratio < 1.0, "All span masking ratios must be between 0.0 and 1.0."
            
            # This mean_length / mask_ratio combo becomes one of the span corruption denoising tasks
            if span_mean_length >= 12 or span_mask_ratio >= 0.3:
                prefix = '[NLG]' # UL2 considers this corruption rate "extreme" and tags it accordingly
            else:
                prefix = '[NLU]' # UL2 considers this corruption rate "regular" and tags it accordingly
            noiser = (
                self.noise_token_sequence,
                dict(mean_span_length=span_mean_length, mask_ratio=span_mask_ratio, prefix=prefix)
            )
            self._noisers.append(noiser)

        # Add "noisers" for the sequential denoising task
        self.sequence_mask_ratios = sequence_mask_ratios
        if self.sequence_mask_ratios is None:
            self.sequence_mask_ratios = []
        elif isinstance(self.sequence_mask_ratios, float):
            self.sequence_mask_ratios = [self.sequence_mask_ratios]
        for sequence_mask_ratio in self.sequence_mask_ratios:
            assert 0 < sequence_mask_ratio < 0.5, "All sequence masking ratios must be between 0.0 and 0.5."

            # This mask_ratio becomes one of the sequential denoising tasks
            noiser = (
                self.noise_token_sequence,
                dict(mean_span_length=None, mask_ratio=span_mask_ratio, prefix='[S2S]')
            )
            self._noisers.append(noiser)

        if not self._noisers:
            raise ValueError("No denoising tasks were included. Make sure to set `span_mean_lengths_and_ratios` and/or `sequence_mask_ratios`.")
        
        self.sentinel_tokens = np.array(tokenizer.additional_special_tokens_ids)

    @staticmethod
    def _sample_mask_array(length: int, mask_ratio: float, mean_span_length: float):
        if mask_ratio == 0.0:
            return np.zeros(length)
        # This first block computes the number of noise/non-noise spans and the total tokens in each.
        # Extra steps are taken to handle edge cases that cause degeneracy.
        starting_length = length
        length = np.maximum(length, 2)
        num_noise_tokens = int(np.round(mask_ratio * float(length)))
        num_noise_tokens = np.minimum(np.maximum(num_noise_tokens, 1), length - 1)
        num_spans = int(np.round(float(num_noise_tokens) / mean_span_length))
        num_noise_spans = np.maximum(num_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # Sample the noise/non-noise span lengths and interleave them to generate the mask array.
        # Note: We always start with a non-noise span.
        def _sample_span_lengths(total_tokens, num_spans):
            """Samples lengths of num_spans segments, the combined length of which equals total_tokens"""
            span_markers = np.less(np.arange(total_tokens - 1), num_spans - 1)[np.random.permutation(total_tokens - 1)]
            span_start_indicator = np.concatenate([[0], span_markers])
            span_id = np.cumsum(span_start_indicator).reshape(-1, 1)
            spans = np.arange(num_spans).reshape(1, -1)
            span_lengths = np.sum(span_id == spans, axis=0)
            return span_lengths

        noise_span_lengths = _sample_span_lengths(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _sample_span_lengths(num_nonnoise_tokens, num_noise_spans)
        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
            [num_noise_spans * 2])
        
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros(length)
        span_start_indicator[span_starts] = 1
        span_id = np.cumsum(span_start_indicator)
        is_noise = np.equal(np.mod(span_id, 2), 1)

        mask = is_noise[:starting_length]

        return mask

    def apply_mask(self, tokens, mask, use_sentinels):
        if not use_sentinels:
            # The logic is simple if we do not mark replaced spans with sentinel tokens
            noised_tokens = np.array(tokens)[np.logical_not(mask)]

            # Ensure there's an end-of-sentence token at the end
            if noised_tokens[-1] != self.tokenizer.eos_token_id:
                noised_tokens = np.concatenate([noised_tokens, [self.tokenizer.eos_token_id]])

            return noised_tokens

        # Masking at previous token
        prev_token_mask = np.concatenate([[0], mask[:-1]])

        # Decompose mask into start-of-span mask and non-start-of-span mask
        start_of_noise_span_token = np.logical_and(mask, np.logical_not(prev_token_mask))
        nonstart_noise_span_token = np.logical_and(mask, prev_token_mask)

        # Replace tokens at the start of each noise span with its corresponding sentinel token
        tokens = np.where(start_of_noise_span_token,
                          self.sentinel_tokens[np.cumsum(start_of_noise_span_token)-1],
                          tokens)

        # Remove masked tokens (but preserving the sentinel tokens)
        noised_tokens = tokens[np.logical_not(nonstart_noise_span_token)]

        # Ensure there's an end-of-sentence token at the end
        if noised_tokens[-1] != self.tokenizer.eos_token_id:
            noised_tokens = np.concatenate([noised_tokens, [self.tokenizer.eos_token_id]])
        return noised_tokens

    def noise_token_sequence(self, example: Mapping[str, Any], mask_ratio: float, mean_span_length: Optional[float], prefix: Optional[str]):
        """Span corruption applicable to all UL2 denoising tasks
        """
        # Extract the raw text tokens (trim if we need to)
        length = sum(example['attention_mask'])
        if length > self.max_seq_length:
            length = self.max_seq_length
        tokens = example['input_ids'][:length]

        # Figure out if there are any prefix tokens to handle
        if prefix is not None:
            if prefix not in self._denoiser_tag_token_ids:
                raise KeyError(f"Prefix {prefix} is not valid. Must be one of: {', '.join(self._denoiser_tag_token_ids.keys())}")
            prefix_tokens = self._denoiser_tag_token_ids[prefix]
        else:
            prefix_tokens  = []

        # mean_span_length==None is a special case for "sequential" denoising (where a single span at the end of the sequence is masked)
        if mean_span_length is None:
            # This ensures that exactly 1 span will be produced and that trimming to max_seq_length will not cut off the sentinel and <EOS>
            min_span_length = np.maximum(1, length + len(prefix_tokens) + 2 - self.max_seq_length)
            max_span_length = np.maximum(min_span_length, np.minimum(length-1, 2*mask_ratio*length))
            mean_span_length = np.floor(np.random.uniform(low=min_span_length, high=max_span_length))
            mask_ratio = mean_span_length / length
            use_sentinels = False
        else:
            use_sentinels = True

        # Generate the mask
        mask = self._sample_mask_array(length, mask_ratio, mean_span_length) # This function can be used for all the UL2 noising functions
        assert mask[0] == 0 # The sequence should always be unmasked at the beginning

        # Generate the input/label sequences given the raw tokens and the mask
        tokens_inputs = self.apply_mask(tokens, mask, use_sentinels)
        tokens_labels = self.apply_mask(tokens, 1-mask, use_sentinels)

        # Tag the inputs with any prefix
        if prefix is not None:
            tokens_inputs = np.concatenate([prefix_tokens, tokens_inputs])

        # Trim if necessary
        if len(tokens_inputs) > self.max_seq_length:
            tokens_inputs = tokens_inputs[:self.max_seq_length]
        if len(tokens_labels) > self.max_seq_length:
            tokens_labels = tokens_labels[:self.max_seq_length]

        tokens_inputs = torch.LongTensor(tokens_inputs)
        tokens_labels = torch.LongTensor(tokens_labels)

        if self.decoder_only_format:
            return self._populate_decoder_only(tokens_inputs, tokens_labels)
        else:
            return self._populate_encoder_decoder(tokens_inputs, tokens_labels)

    def _populate_encoder_decoder(self, tokens_inputs: torch.LongTensor, tokens_labels: torch.LongTensor):
        example = {}
        # Re-populate with an empty, padded example
        example['input_ids'] = torch.full((self.max_seq_length,), self.tokenizer.pad_token_id, dtype=torch.int32) 
        example['labels']    = torch.full((self.max_seq_length,), -100, dtype=torch.int32) # Note: The HF code is hardcoded to use -100 as the ignore index
        example['attention_mask'] = torch.zeros_like(example['input_ids'])
        example['decoder_attention_mask'] = torch.zeros_like(example['labels'])

        # Fill in with the processed results
        example['input_ids'][:len(tokens_inputs)] = tokens_inputs
        example['labels'][:len(tokens_labels)] = tokens_labels
        example['attention_mask'][:len(tokens_inputs)] = 1
        example['decoder_attention_mask'][:len(tokens_labels)] = 1
        return example

    def _populate_decoder_only(self, tokens_inputs: torch.LongTensor, tokens_labels: torch.LongTensor):
        example = {}
        # Re-populate with an empty, padded example
        example['input_ids'] = torch.full((self.max_seq_length*2,), self.tokenizer.pad_token_id, dtype=torch.int32) 
        example['labels']    = torch.full((self.max_seq_length*2,), -100, dtype=torch.int32) # Note: -100 is often hardcoded as the ignore index
        example['attention_mask'] = torch.full((self.max_seq_length*2,), 0, dtype=torch.bool)
        example['bidirectional_mask'] = torch.full((self.max_seq_length*2,), 0, dtype=torch.bool)

        n_input = len(tokens_inputs)
        n_label = len(tokens_labels)
        n_concat = n_input + n_label
        assert n_concat <= self.max_seq_length * 2

        tokens_concat = torch.concat([tokens_inputs, tokens_labels], dim=0)

        # Fill in with the processed results
        example['input_ids'][:n_concat] = tokens_concat
        # (Labels are a shifted version of `input_ids`, with the portion belonging to `tokens_inputs` masked so no loss is generated from them.)
        example['labels'][:n_concat-1] = tokens_concat[1:]
        example['labels'][:n_input-1] = -100
        example['attention_mask'][:n_concat-1] = 1
        example['bidirectional_mask'][:n_input] = 1
        return example

    def __call__(self, examples: List[Dict[str, Any]]):
        """Batch examples processed by the span corrupter."""
        processed_examples = []
        for example in examples:
            noiser_fcn, noiser_kwargs = random.choice(self._noisers)
            processed_examples.append(noiser_fcn(example, **noiser_kwargs))
        batch = self.tokenizer.pad(processed_examples)
        
        # Truncate portions of the inputs that are purely padding (up to a multiple of 8)
        multiple_of = 8
        n_examples_per_length = batch['attention_mask'].sum(0)
        keep_tokens = torch.sum(n_examples_per_length > 0)
        keep_tokens = int(multiple_of * torch.ceil(keep_tokens / multiple_of))
        batch['input_ids'] = batch['input_ids'][:, :keep_tokens]
        batch['attention_mask'] = batch['attention_mask'][:, :keep_tokens]
        if self.decoder_only_format:
            batch['labels'] = batch['labels'][:, :keep_tokens]
            batch['bidirectional_mask'] = batch['bidirectional_mask'][:, :keep_tokens]
        
        else:
            # Truncate portions of the decoder inputs that are purely padding
            n_examples_per_length = batch['decoder_attention_mask'].sum(0)
            keep_tokens = n_examples_per_length > 0
            batch['labels'] = batch['labels'][:, :keep_tokens]
            batch['decoder_attention_mask'] = batch['decoder_attention_mask'][:, :keep_tokens]
        
        return batch
    

def build_c4_dataloader(cfg: Mapping[str, Any], device_batch_size: int):

    assert cfg.name == 'c4', f'Tried to build c4 dataloader with cfg.name={cfg.name}'
    dataset = StreamingC4(split=cfg.dataset.split,
                            remote=cfg.dataset.remote,
                            local=cfg.dataset.local,
                            shuffle=cfg.dataset.shuffle,
                            prefetch=cfg.dataset.prefetch,
                            tokenizer_name=cfg.dataset.tokenizer_name,
                            max_seq_len=cfg.dataset.max_seq_len,
                            group_method=cfg.dataset.group_method,
                            batch_size=device_batch_size)

    collate_fn = MixtureOfDenoisersCollator(
        tokenizer=dataset.tokenizer,
        max_seq_length=cfg.dataset.max_seq_len,
        decoder_only_format=cfg.decoder_only_format,
        span_mean_lengths_and_ratios=cfg.mixture_of_denoisers.get('span_mean_lengths_and_ratios', None),
        sequence_mask_ratios=cfg.mixture_of_denoisers.get('sequence_mask_ratios', None),
    )

    return DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=device_batch_size,
        drop_last=cfg.drop_last,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=cfg.persistent_workers,
        timeout=cfg.timeout,
    )

# Helpful to test if your dataloader is working locally
# Run `python data.py [remote] [local, optional]` and verify that batches are printed out
if __name__ == '__main__':
    remote = sys.argv[1]
    if len(sys.argv) > 2:
        local = sys.argv[2]
    else:
        local = remote
    print (f'Reading val split from {remote} -> {local}')

    cfg = {
        'name': 'c4',
        'dataset': {
            'remote': remote,
            'local': local,
            'split': 'val',
            'shuffle': True,
            'prefetch': 1000,
            'tokenizer_name': 't5-base',
            'max_seq_len': 128,
            'group_method': 'concat',
        },
        'mixture_of_denoisers': {
            'span_mean_lengths_and_ratios': [[3, .15], [12, .15]],
            'sequence_mask_ratios': 0.25,
        },
        'drop_last': False,
        'num_workers': 0, #4,
        'pin_memory': True,
        'prefetch_factor': 2,
        'persistent_workers': False, #True,
        'timeout': 0, #30,
    }
    cfg = om.create(cfg)
    device_batch_size = 2

    loader = build_c4_dataloader(cfg, device_batch_size)
    tokenizer = loader.dataset.tokenizer
    for batch_ix, batch in enumerate(islice(loader, 5)):
        print('\n')
        print ('#'*20, f'Batch {batch_ix}', '#'*20)
        for k, v in batch.items():
            print (k, v.shape, v.dtype)
        for sample_ix, token_sample in enumerate(batch['input_ids']):
            labels = batch['labels'][sample_ix]
            attn_inputs = batch['attention_mask'][sample_ix].to(torch.bool)
            attn_labels = batch['decoder_attention_mask'][sample_ix].to(torch.bool)
            print ('-'*20, f' Sample {sample_ix} ', '-'*20)
            print ('Input:  ', tokenizer.decode(token_sample[attn_inputs]))
            print ('Target: ', tokenizer.decode(labels[attn_labels]))

