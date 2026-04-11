from __future__ import annotations

import re
from functools import lru_cache

from transformers import AutoTokenizer

_FALLBACK_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@lru_cache(maxsize=4)
def get_tokenizer(model_name: str):
    try:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        return None


def encode_text(model_name: str, text: str) -> list[int | str]:
    tokenizer = get_tokenizer(model_name)
    if tokenizer is None:
        return _FALLBACK_TOKEN_RE.findall(text or "")
    return list(tokenizer.encode(text or "", add_special_tokens=False))


def decode_tokens(model_name: str, token_ids: list[int | str]) -> str:
    if not token_ids:
        return ""
    if isinstance(token_ids[0], str):
        return " ".join(str(token) for token in token_ids).strip()
    tokenizer = get_tokenizer(model_name)
    if tokenizer is None:
        return " ".join(str(token) for token in token_ids).strip()
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()


def count_tokens(model_name: str, text: str) -> int:
    return len(encode_text(model_name, text))
