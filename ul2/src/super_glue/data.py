"""This is a preliminary implementation of SuperGLUE for EncDec"""

import datasets
import transformers
import logging
import torch

from typing import Mapping, Any, Optional, List
from composer.utils import dist
from torch.utils.data import DataLoader
from torchmetrics import Metric

log = logging.getLogger(__name__)


_task_prefix_column_names = {
    'boolq': ['passage', 'question'],
    'cb': ['hypothesis', 'premise'],
    'copa': ['choice1', 'choice2', 'premise', 'question'],
    'multirc': ['question', 'answer', 'paragraph'],
    'record': ['passage', 'query'],
    'rte': ['hypothesis', 'premise'],
    'wic': ['sentence1', 'sentence2', 'word'],
    'wsc': ['text', 'span1_text', 'span2_text'],
}
_task_target_maps = {
    'boolq': {0: 'False', 1: 'True'},
    'cb': {0: 'entailment', 1: 'not_entailment', 2: 'neutral'},
    'copa': {0: 'choice1', 1: 'choice2'},
    'multirc': {0: 'False', 1: 'True'},
    'record': lambda example: '' if len(example['answers'])==0 else example['answers'][0],
    'rte': {0: 'entailment', 1: 'not_entailment', 2: 'neutral'},
    'wic': {0: 'False', 1: 'True'},
    'wsc': {0: 'False', 1: 'True'},
}

def create_super_glue_dataset(
    task: str,
    tokenizer_name: str,
    split: str,
    max_seq_length: int = 256,
    max_retries: int = 10,
    num_workers: int = 0,
    extra_prefix: Optional[str] = None,
):
    if task not in _task_prefix_column_names:
        raise ValueError(f'task ({task}) must be one of {_task_prefix_column_names.keys()}')
        
    if (max_seq_length % 8) != 0:
        log.warning('For performance, a max_seq_length as a multiple of 8 is recommended.')

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=max_seq_length)  #type: ignore (thirdparty)

    log.info(f'Loading {task.upper()} on rank {dist.get_global_rank()}')
    download_config = datasets.DownloadConfig(max_retries=max_retries)
    dataset = datasets.load_dataset(
        'super_glue',
        task,
        split=split,
        download_config=download_config,
    )

    log.info(f'Starting tokenization by preprocessing over {num_workers} threads!')
    prefix_column_order = _task_prefix_column_names[task]
    target_extractor_map = _task_target_maps[task]
    if isinstance(target_extractor_map, dict):
        target_extractor = lambda example: target_extractor_map[example['label']]
    else:
        assert callable(target_extractor_map)
        target_extractor = target_extractor_map

    def convert_padding_token(input_ids: List, attention_mask: List, new_pad_id: int=-100):
        n_non_padded = sum(attention_mask)
        n_total = len(input_ids)
        
        if n_non_padded == n_total:
            return input_ids

        non_padded_input_ids = input_ids[:n_non_padded]
        return non_padded_input_ids + ([new_pad_id] * (n_total - n_non_padded))

    def tokenize_function(inp: Mapping[str, Any]):
        # truncates sentences to max_length or pads them to max_length
        if extra_prefix is not None:
            prefix = f'{extra_prefix} {task}'
        else:
            prefix = task
        text = [prefix]
        for prefix_column in prefix_column_order:
            text.append(f'{prefix_column}: {inp[prefix_column]}')
        encoder_text = '\n'.join(text)
        
        decoder_text = target_extractor(inp)

        encoder_dict = tokenizer(
            text=encoder_text,
            padding='max_length',
            max_length=max_seq_length,
            truncation=True,
        )

        decoder_dict = tokenizer(
            text=decoder_text,
            padding='max_length',
            max_length=20,
            truncation=True,
        )

        # HF hardcodes the ignore_index to -100 in the loss, so ensure that
        # the padding id is -100 for the loss-generating batch keys
        decoder_dict['input_ids'] = convert_padding_token(decoder_dict['input_ids'], decoder_dict['attention_mask'], new_pad_id=-100)

        return {
            'input_ids': encoder_dict.input_ids,
            'attention_mask': encoder_dict.attention_mask,
            'labels': decoder_dict.input_ids,
            'decoder_attention_mask': decoder_dict.attention_mask,
        }
    
    assert isinstance(dataset, datasets.Dataset)
    columns_to_remove = list(dataset[0].keys())
    dataset = dataset.map(
        tokenize_function,
        batched=False,
        num_proc=None if num_workers == 0 else num_workers,
        # batch_size=3, #1000,
        remove_columns=columns_to_remove,
        new_fingerprint=f'{task}-{tokenizer_name}-tokenization-{max_seq_length}-{split}',
        load_from_cache_file=False,
    )
    return dataset

def build_super_glue_task_dataloader(cfg: Mapping[str, Any], device_batch_size: int):

    assert cfg.name == 'super_glue', f'Tried to build super_glue dataloader with cfg.name={cfg.name}'
    dataset = create_super_glue_dataset(cfg.dataset.task, 
                                        cfg.dataset.tokenizer_name,
                                        cfg.dataset.split,
                                        cfg.dataset.max_seq_length,
                                        cfg.dataset.get('extra_prefix', None))

    return DataLoader(
        dataset,
        collate_fn=transformers.default_data_collator,
        batch_size=device_batch_size,
        sampler=dist.get_sampler(dataset, drop_last=cfg.drop_last, shuffle=cfg.shuffle),
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=cfg.persistent_workers,
        timeout=cfg.timeout,
    )

class ExactMatch(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    def __init__(self, ignore_index: Optional[int]=-100):
        super().__init__()
        self.ignore_index = ignore_index
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    @staticmethod
    def _input_format(preds: torch.Tensor, target: torch.Tensor):
        if preds.ndim == target.ndim:
            assert preds.shape == target.shape
            return preds, target

        else:
            preds = preds.argmax(-1)
            assert preds.shape == target.shape
            return preds, target

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = self._input_format(preds, target)

        if self.ignore_index is not None:
            token_match = torch.logical_or(preds == target, target == self.ignore_index)
        else:
            token_match = preds == target
        exact_match = torch.all(token_match, dim=-1)

        self.correct += torch.sum(exact_match)
        self.total += exact_match.numel()

    def compute(self):
        return self.correct.float() / self.total