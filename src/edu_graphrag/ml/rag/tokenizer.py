"""Lexical tokenizers for BM25 (Cyrillic + Latin, no external NLP dependencies).

`word`   - lowercase alphanumeric runs of 2+ chars, `ё` folded into its plain form.
`prefixN` - the same tokens truncated to N characters: a crude stemmer that maps Russian
            inflections (`матрица`, `матрицы`, `матрицей`) to one term.
"""

import re
from collections.abc import Callable

_TOKEN_RE = re.compile(r"[^\W_]{2,}")
_PREFIX_RE = re.compile(r"^prefix(\d+)$")
# Cyrillic "yo" and "ye" by code point (the literal letters trip ruff's ambiguous-char rule).
_YO, _YE = chr(0x451), chr(0x435)

Tokenizer = Callable[[str], list[str]]


def tokenize(text: str) -> list[str]:
    """`"Матрица Грама, x_1"` -> `["матрица", "грама"]`; `ё` is folded like in search engines."""
    return _TOKEN_RE.findall(text.lower().replace(_YO, _YE))


def prefix_tokenize(text: str, length: int) -> list[str]:
    return [token[:length] for token in tokenize(text)]


def get_tokenizer(name: str) -> Tokenizer:
    """Resolve a tokenizer by its logged name (`word` or `prefixN`)."""
    if name == "word":
        return tokenize
    match = _PREFIX_RE.match(name)
    if match and int(match.group(1)) > 0:
        length = int(match.group(1))
        return lambda text: prefix_tokenize(text, length)
    raise ValueError(f"unknown tokenizer: {name!r}")
