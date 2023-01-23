from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union

def adapt_tokenizer_for_denoising(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], num_sentinel_tokens: int=100):
    """Adds sentinel tokens to the tokenizer (if they are missing) and sets the padding token (if it is missing)."""
    # Add sentinel tokens (e.g., <extra_id_0>, <extra_id_1>, and so on). Has no effect if these are already in the vocab.
    assert 0 <= num_sentinel_tokens <= 1000
    sentinels_to_add = [f'<extra_id_{i}>' for i in range(num_sentinel_tokens)]
    tokenizer.add_tokens(sentinels_to_add, special_tokens=True)

    # If the padding token has not been set, add <pad> and use it
    if tokenizer.pad_token is None:
        tokenizer.add_tokens('<pad>', special_tokens=True)
        tokenizer.pad_token = '<pad>'
        assert tokenizer.pad_token_id is not None
