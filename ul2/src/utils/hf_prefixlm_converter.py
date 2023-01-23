from typing import List, Optional, Union
from types import MethodType

import torch

from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM
from transformers.models.gptj.modeling_gptj import GPTJForCausalLM

_SUPPORTED_HF_MODELS = (
    GPT2LMHeadModel,
    GPTJForCausalLM,
    GPTNeoForCausalLM,
    GPTNeoXForCausalLM,
)

CAUSAL_GPT_TYPES = Union[
    GPT2LMHeadModel,
    GPTJForCausalLM,
    GPTNeoForCausalLM,
    GPTNeoXForCausalLM,
]

def convert_gpt_causal_lm_to_prefix_lm(model: CAUSAL_GPT_TYPES) -> CAUSAL_GPT_TYPES:
    """Converts a GPT-style Causal LM to a Prefix LM.
    
    Supported HuggingFace model classes:
        - `GPT2LMHeadModel`
        - `GPTNeoForCausalLM`
        - `GPTNeoXForCausalLM`
        - `GPTJForCausalLM`

    Conversion to a Prefix LM is done by wrapping the `forward` and `generate` methods.
    To preserve the API, the original methods are renamed to `_original_forward` and
    `_original_generate`, and replaced with new `forward` and `generate` methods that wrap 
    them, respectively.

    Notes on `forward` method conversion:

        After conversion, the `forward` method will handle a new input, `bidirectional_mask`,
        which should be a [batch_size, seq_length] byte tensor, where 1 indicates token positions
        belonging to the prefix (prefix tokens can attend to one another bidirectionally), and 
        0 indicates token positions belonging to the target.

        The new `forward` method will incorporate `bidirectional_mask` (if supplied) into the existing
        causal mask, call the original `forward` method, and reset the causal masks before returning
        the result.

    Notes on `generate` method conversion:

        After conversion, the `generate` method will have the same signature but will internally
        convert all causal masks to be purely bidirectional, call the original `generate` method, and
        reset the causal masks before returning the result.

        This works thanks to the logic of the HuggingFace `generate` API, which first encodes the token
        "prompt" passed to `generate` (which is treated as the prefix) and then sequentially generates
        each new token. Encodings are cached as generation happens, so all prefix tokens can attend to one 
        another (as expected in a Prefix LM) and generated tokens can only attend to prefix tokens and
        previously-generated tokens (also as expected in a Prefix LM).

    Notes on training:

        To actually train the converted model as a Prefix LM, training batches will need to indicate
        the prefix/target structure by including `bidirectional_mask` as part of the batch inputs.
        
        **This is not a standard input and requires custom layers either within or after your dataloader.**

        In addition to adding `bidirectional_mask` to the batch, this custom code should modify `labels`
        such that `batch['labels'][batch['bidirectional_mask'] == 1] == -100`.
        That is, the prefix portion of the sequence should not generate any loss. Loss should only be
        generated by the target portion of the sequence.

    Notes on `GPTNeoForCausalLM`:

        To simplify the implementation, "global" and "local" attention layers are handled differently.
        For "global" layers, we handle conversion as described above. For "local" layers, which use a 
        causal attention mask within a restricted local window, we do not alter the masking.
    """
    
    if hasattr(model, '_prefix_lm_converted'):
        return model
    
    assert isinstance(model, _SUPPORTED_HF_MODELS)
    assert model.config.add_cross_attention == False, "Only supports decoder-only models"

    # Rename methods to allow:
    #  - new `forward` to wrap original `forward`
    #  - new `generate` to wrap original `generate`
    setattr(model, '_original_forward', getattr(model, 'forward'))
    setattr(model, '_original_generate', getattr(model, 'generate'))

    def forward(self: CAUSAL_GPT_TYPES, *args, bidirectional_mask: Optional[torch.ByteTensor]=None, **kwargs):
        """Wrapper around original `forward()` that enables PrefixLM-style attention."""
        if bidirectional_mask is None:
            # This wrapper is a no-op if bidirectional masks are not supplied
            return self._original_forward(*args, **kwargs) # type: ignore
        
        
        attn_modules = _get_attn_modules(model)

        # Handle bidirectional_mask sizing
        b, s = bidirectional_mask.shape
        _, _, _, max_length = attn_modules[0].bias.shape # Note: all attn_modules.bias have the same size
        assert s <= max_length
        if s < max_length:
            pad = torch.zeros((b, max_length-s), dtype=bidirectional_mask.dtype, device=bidirectional_mask.device)
            bidirectional_mask = torch.cat([bidirectional_mask, pad], dim=1)
        bidirectional = bidirectional_mask.unsqueeze(1).unsqueeze(1)

        # Incorporate the bidirectional mask into the original causal mask
        for attn_module in attn_modules:
            attn_module.bias.data = torch.logical_or(attn_module.bias.data, bidirectional) # type: ignore
        
        # Collect outputs using the model's original forward method
        output = self._original_forward(*args, **kwargs) # type: ignore

        # Reset the masks
        for attn_module in attn_modules:
            attn_module.bias.data = torch.tril(attn_module.bias.data[0, 0])[None, None] # type: ignore
        
        # Return the outputs
        return output

    def generate(self: CAUSAL_GPT_TYPES, *args, **kwargs):
        """Wrapper around original `generate()` that enables PrefixLM-style attention."""

        attn_modules = _get_attn_modules(model)

        # A convenient answer to PrefixLM generation is to set the causal mask to be bidirectional.
        # All the tokens in the input prompt can attend to one another and, since tokens are generated
        # one-by-one, each new token gets to see everything behind it.
        # This depends on activations being cached and not updated, which is how the HF implementation works.
        for attn_module in attn_modules:
            attn_module.bias.data[:] = 1 # type: ignore

        # Collect outputs using the model's original forward method
        output = self._original_generate(*args, **kwargs) # type: ignore

        # Reset the masks
        for attn_module in attn_modules:
            attn_module.bias.data = torch.tril(attn_module.bias.data[0, 0])[None, None] # type: ignore
        
        # Return the outputs
        return output

    # Replace `forward` and `generate` with the new wrappers
    setattr(model, 'forward', MethodType(forward, model))
    setattr(model, 'generate', MethodType(generate, model))

    # Finally, tag the model so that this conversion cannot happen again.
    setattr(model, '_prefix_lm_converted', True)
    return model

def _get_attn_modules(model: CAUSAL_GPT_TYPES) -> List[torch.nn.Module]:
    """Gets a list of the model's attention modules.
    
    Each module has a `bias` buffer used for causal masking. The Prefix LM
    conversion adds logic to dynamically manipulate these biases to support
    Prefix LM attention masking.
    """
    attn_modules = []

    if isinstance(model, GPTNeoXForCausalLM):
        blocks = model.gpt_neox.layers
    else:
        blocks = model.transformer.h

    for block in blocks: # type: ignore
        if isinstance(model, GPTNeoForCausalLM):
            # Ignore "local" layers in this model type
            if block.attn.attention_type != 'global':
                continue
            attn_module = block.attn.attention
        elif isinstance(model, GPTNeoXForCausalLM):
            attn_module = block.attention
        else:
            attn_module = block.attn

        attn_modules.append(attn_module)

    return attn_modules