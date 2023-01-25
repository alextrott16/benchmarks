from __future__ import annotations

from typing import Optional

from composer.metrics.nlp import (LanguageCrossEntropy,
                                  MaskedAccuracy)
from composer.models.huggingface import HuggingFaceModel
from composer.utils.import_helpers import MissingConditionalImportError

from src import utils

__all__ = ['create_hf_prefix_lm']


def create_hf_prefix_lm(pretrained_model_name: str = 'gpt2',
                        tokenizer_name: str = 'gpt2',
                        use_pretrained: Optional[bool] = False,
                        model_config: Optional[dict] = None,
                        gradient_checkpointing: Optional[bool] = False):
    """...

    For more information, see `Transformers <https://huggingface.co/transformers/>`_.

    Args:
        pretrained_model_name (str): Name of the Hugging Face model to instantiate. Default: ``'gpt2'``.
        tokenizer_name (str): Tokenizer name used to preprocess the dataset and validate the models inputs. To be compatible
            with denoising tasks, special sentinel tokens ("<extra_id_0>", "<extra_id_1>", etc.) will be added. The model's vocab_size
            will be hardcoded to match the resulting vocab_size of the tokenizer.
        use_pretrained (bool, optional): Whether to initialize the model with the pretrained weights. Default: ``False``.
        model_config (dict, optional): The settings used to create a Hugging Face BertConfig. BertConfig is used to specify
            the architecture of a Hugging Face model.
        gradient_checkpointing (bool, optional): Use gradient checkpointing. Default: ``False``.
    """
    try:
        import transformers
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='nlp',
                                            conda_package='transformers') from e

    # Set up the tokenizer (add tokens for denoising sentinels -- has no effect if the tokenizer already has them)
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    utils.adapt_tokenizer_for_denoising(tokenizer, num_sentinel_tokens=100)
    new_vocab_size = len(tokenizer)

    if not model_config:
        model_config = {}

    if not pretrained_model_name:
        pretrained_model_name = 'gpt2'

    if use_pretrained:
        assert transformers.AutoModelForCausalLM.from_pretrained is not None, 'AutoModelForCausalLM has from_pretrained method'
        model = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name, **model_config)
    else:
        config = transformers.AutoConfig.from_pretrained(
            pretrained_model_name, **model_config)
        assert transformers.AutoModelForCausalLM.from_config is not None, 'AutoModelForCausalLM has from_config method'
        model = transformers.AutoModelForCausalLM.from_config(config)

    # Convert the Causal LM into a Prefix LM via our custom, lightweight wrapper
    model = utils.convert_hf_causal_lm_to_prefix_lm(model)

    # Expand the embeddings/vocab size to match the tokenizer (which has possibly just had new tokens added above)
    if model.config.vocab_size != new_vocab_size:
        model.resize_token_embeddings(new_num_tokens=new_vocab_size)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()  # type: ignore

    metrics = [
        LanguageCrossEntropy(ignore_index=-100,
                             vocab_size=model.config.vocab_size),
        MaskedAccuracy(ignore_index=-100)
    ]
    return HuggingFaceModel(model=model,
                            tokenizer=tokenizer,
                            use_logits=True,
                            metrics=metrics)