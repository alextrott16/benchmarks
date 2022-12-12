# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Hugging Face BERT wrapped inside a :class:`.ComposerModel`."""

from __future__ import annotations

from typing import Optional

from composer.metrics.nlp import LanguageCrossEntropy, MaskedAccuracy #, BinaryF1Score
from composer.models.huggingface import HuggingFaceModel
from composer.utils.import_helpers import MissingConditionalImportError


# from torchmetrics import MeanSquaredError
# from torchmetrics.classification.accuracy import Accuracy
# from torchmetrics.classification.matthews_corrcoef import MatthewsCorrCoef
# from torchmetrics.regression.spearman import SpearmanCorrCoef

__all__ = ['create_hf_t5']

def create_hf_t5(pretrained_model_name: str = 't5-base',
                 use_pretrained: Optional[bool] = False,
                 model_config: Optional[dict] = None,
                 tokenizer_name: Optional[str] = None):
    """T5 model based on |:hugging_face:| Transformers.

    For more information, see `Transformers <https://huggingface.co/transformers/>`_.

    Args:
        pretrained_model_name (str): Name of the Hugging Face model to instantiate. Default: ``'t5-base'``.
        use_pretrained (bool, optional): Whether to initialize the model with the pretrained weights. Default: ``False``.
        model_config (dict): The settings used to create a Hugging Face T5Config. T5Config is used to specify the
            architecture of a Hugging Face model.
        tokenizer_name (str, optional): Tokenizer name used to preprocess the dataset and validate the models inputs.

   To create a |:hugging_face:| T5 model for Masked Language Model pretraining:

    .. testcode::

        from src.hf_t5 import create_hf_t5
        model = create_hf_t5()

    """
    try:
        import transformers
        from transformers import T5ForConditionalGeneration, T5Config
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='transformers') from e

    if not pretrained_model_name:
        pretrained_model_name = 't5-base'

    # setup the tokenizer
    if tokenizer_name:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name)

    if not model_config:
        model_config = {}

    if use_pretrained:
        model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=pretrained_model_name, **model_config)
    else:
        config = T5Config.from_pretrained(pretrained_model_name, **model_config)
        model = T5ForConditionalGeneration(config)

    # We use `len(tokenizer) instead of `model.config.vocab_size` because `HuggingFaceModel` will set the latter to the former
    metrics = [
        LanguageCrossEntropy(ignore_index=-100, vocab_size=len(tokenizer)), # Note: The HF code is hardcoded to use -100 as the ignore index
        MaskedAccuracy(ignore_index=-100)
    ]
    return HuggingFaceModel(model=model, tokenizer=tokenizer, use_logits=True, metrics=metrics)

