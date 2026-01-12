"""Utility helpers for normalization and parsing."""
from __future__ import annotations

import re
from typing import Iterable, List, Optional
from dateutil import parser as date_parser


NON_WORD_RE = re.compile(r"[^A-Za-z0-9]+")
CURRENCY_RE = re.compile(r"^[\$€£¥]?\(?-?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?$")
CURRENCY_SYMBOLS_RE = re.compile(r"[\$€£¥]|AED|USD|EUR|GBP|JPY", re.IGNORECASE)


def normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def tokenize_header(value: str) -> List[str]:
    cleaned = NON_WORD_RE.sub(" ", value.lower()).strip()
    return [token for token in cleaned.split() if token]


def tokenize_text(value: str) -> List[str]:
    cleaned = NON_WORD_RE.sub(" ", value.lower()).strip()
    return [token for token in cleaned.split() if token]


def jaccard_similarity(a: Iterable[str], b: Iterable[str]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def parse_number(value: str) -> Optional[float]:
    if not value:
        return None
    cleaned = value.strip()
    negative = cleaned.startswith("(") and cleaned.endswith(")")
    if negative:
        cleaned = cleaned[1:-1]
    cleaned = CURRENCY_SYMBOLS_RE.sub("", cleaned)
    cleaned = cleaned.replace(",", "").strip()
    if negative and cleaned and not cleaned.startswith("-"):
        cleaned = f"-{cleaned}"
    try:
        return float(cleaned)
    except ValueError:
        return None


def is_currency(value: str) -> bool:
    if not value:
        return False
    return bool(CURRENCY_RE.match(value.strip()))


def is_number(value: str) -> bool:
    if not value:
        return False
    return parse_number(value) is not None


def is_date(value: str) -> bool:
    if not value:
        return False
    text = value.strip()
    if not text:
        return False
    has_alpha = any(ch.isalpha() for ch in text)
    has_sep = any(sep in text for sep in ("/", "-", "."))
    if not has_alpha and not has_sep:
        return False
    try:
        date_parser.parse(text, fuzzy=False)
        return True
    except (ValueError, TypeError):
        return False


def safe_join(parts: List[str]) -> str:
    return " ".join([p for p in (part.strip() for part in parts) if p])


def normalize_numeric(value: str) -> str:
    if not value:
        return value
    original = value.strip()
    if not original:
        return original
    cleaned = original
    negative = cleaned.startswith("(") and cleaned.endswith(")")
    if negative:
        cleaned = cleaned[1:-1]
    cleaned = CURRENCY_SYMBOLS_RE.sub("", cleaned)
    cleaned = cleaned.replace(",", "").strip()
    if negative and cleaned and not cleaned.startswith("-"):
        cleaned = f"-{cleaned}"
    if not cleaned:
        return original
    try:
        num = float(cleaned)
    except ValueError:
        return original
    if "." in cleaned:
        decimals = len(cleaned.split(".", 1)[1])
        return f"{num:.{decimals}f}"
    if num.is_integer():
        return str(int(num))
    return str(num)
