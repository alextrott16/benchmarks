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
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
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
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
        span_mean_lengths_and_ratios: Optional[Union[List[List[float]], List[float]]] = None,
        sequence_mask_ratios: Optional[Union[List[float], float]] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self._denoiser_tags = [
            '[R]', # "Regular" span corruption
            '[X]', # "Extreme" span corruption
            '[S]', # Sequential denoising
        ]
        self._denoiser_tag_token_ids = {} # To be appended to `input_ids` when corrupting
        for tag in self._denoiser_tags:
            tag_tokens = self.tokenizer(tag).input_ids
            special = self.tokenizer.get_special_tokens_mask(tag_tokens, already_has_special_tokens=True)
            # non_special = torch.logical_not(torch.BoolTensor(special))
            self._denoiser_tag_token_ids[tag] = [tag_token for tag_token, spc in zip(tag_tokens, special) if not spc]#tag_tokens[non_special]

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
            noiser = (
                self.span_corrupt_token_sequence,
                dict(mean_span_length=span_mean_length, mask_ratio=span_mask_ratio)
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
                self.sequence_corrupt_token_sequence,
                dict(mask_ratio=sequence_mask_ratio)
            )
            self._noisers.append(noiser)

        if not self._noisers:
            raise ValueError("No denoising tasks were included. Make sure to set `span_mean_lengths_and_ratios` and/or `sequence_mask_ratios`.")
        
        self.sentinel_tokens = tokenizer.additional_special_tokens_ids
        # self.sentinel_tokens = torch.LongTensor(tokenizer.additional_special_tokens_ids)

    def span_corrupt_token_sequence(self, example: Mapping[str, Any], mean_span_length: float, mask_ratio: float):
        tokens = example['input_ids']
        
        # Re-populate with an empty, padded example
        example['input_ids'] = torch.full((self.max_seq_length,), self.tokenizer.pad_token_id) 
        example['labels']    = torch.full((self.max_seq_length,), -100) # Note: The HF code is hardcoded to use -100 as the ignore index
        example['attention_mask'] = torch.zeros_like(example['input_ids'])
        example['decoder_attention_mask'] = torch.zeros_like(example['labels'])

        length = len(tokens)
        if mean_span_length >= 12 or mask_ratio >= 0.3:
            tag = '[X]' # UL2 considers this corruption rate "extreme" and tags it accordingly
        else:
            tag = '[R]' # UL2 considers this corruption rate "regular" and tags it accordingly
        tag_token_ids = self._denoiser_tag_token_ids[tag]
        
        mask_hit = False
        while not mask_hit: # Better to do this multiple times than waste a sample by not corrupting anything
            span_index = 0
            idx_inputs = []
            idx_labels = []
            idx = 0
            # The below procedure implicitly breaks the token sequence into spans and then masks them with a probability of `mask_ratio`
            idx_inputs += tag_token_ids
            while idx < length:
                span = int(np.maximum(2.0, np.round(np.random.normal(mean_span_length, 1))))
                # Begin a masked new span
                if np.random.rand() <= mask_ratio:
                    mask_hit = True
                    # Mark the span
                    idx_inputs += [self.sentinel_tokens[span_index]]
                    idx_labels += [self.sentinel_tokens[span_index]]
                    span_index = min(span_index + 1, len(self.sentinel_tokens) - 1)
                    
                    # Add the tokens to the labels
                    idx_end = min(idx + span, length)
                    idx_labels += tokens[idx:idx_end]
                    idx = idx_end
                    
                    # Don't allow the next token to be part of a masked span
                    if idx < length:
                        idx_inputs += [tokens[idx]]
                        idx = idx + 1
                    
                # Begin a new unmasked span
                else:
                    # Add the tokens to the inputs
                    idx_end = min(idx + span, length)
                    idx_inputs += tokens[idx:idx_end]
                    idx = idx_end
            idx_labels += [self.sentinel_tokens[span_index]]
                    
            assert idx == length

        tokens_inputs = torch.LongTensor(idx_inputs)
        tokens_labels = torch.LongTensor(idx_labels)

        if len(tokens_inputs) > self.max_seq_length:
            tokens_inputs = tokens_inputs[:self.max_seq_length]
        if len(tokens_labels) > self.max_seq_length:
            tokens_labels = tokens_labels[:self.max_seq_length]

        example['input_ids'][:len(tokens_inputs)] = tokens_inputs
        example['labels'][:len(tokens_labels)] = tokens_labels
        example['attention_mask'][:len(tokens_inputs)] = 1
        example['decoder_attention_mask'][:len(tokens_labels)] = 1
        
        return example

    def sequence_corrupt_token_sequence(self, example: Mapping[str, Any], mask_ratio: float):
        tokens = example['input_ids']
        
        # Re-populate with an empty, padded example
        example['input_ids'] = torch.full((self.max_seq_length,), self.tokenizer.pad_token_id)
        example['labels']    = torch.full((self.max_seq_length,), -100) # Note: The HF code is hardcoded to use -100 as the ignore index
        example['attention_mask'] = torch.zeros_like(example['input_ids'])
        example['decoder_attention_mask'] = torch.zeros_like(example['labels'])
        
        length = len(tokens)
        tag_tokens = self._denoiser_tag_token_ids['[S]']
        label_token_length = int(np.round(np.random.uniform(low=1+len(tag_tokens), high=int(length*2*mask_ratio))))
        input_token_length = length - label_token_length
        
        tokens_inputs = torch.LongTensor(tag_tokens + tokens[:input_token_length] + self.sentinel_tokens[:1])
        tokens_labels = torch.LongTensor(self.sentinel_tokens[:1] + tokens[input_token_length:])

        example['input_ids'][:len(tokens_inputs)] = tokens_inputs
        example['labels'][:len(tokens_labels)] = tokens_labels
        example['attention_mask'][:len(tokens_inputs)] = 1
        example['decoder_attention_mask'][:len(tokens_labels)] = 1
        
        return example

    def __call__(self, examples: List[Dict[str, Any]]):
        """Batch examples processed by the span corrupter."""
        processed_examples = []
        for example in examples:
            noiser_fcn, noiser_kwargs = random.choice(self._noisers)
            processed_examples.append(noiser_fcn(example, **noiser_kwargs))
        batch = self.tokenizer.pad(processed_examples)
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

