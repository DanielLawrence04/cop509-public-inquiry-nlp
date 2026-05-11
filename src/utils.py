"""
Shared utility functions used across the pipeline.
"""

import re
import json
import unicodedata
from pathlib import Path


def clean_text(text: str) -> str:
    """Remove excessive whitespace and normalise unicode characters."""
    text = text.replace("\u00ad", "")
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation for comparison tasks."""
    text = clean_text(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def token_overlap_score(a: str, b: str) -> float:
    """
    Jaccard-style token overlap between two strings.
    Returns a score in [0, 1].
    """
    tokens_a = set(normalize_text(a).split())
    tokens_b = set(normalize_text(b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def save_json(data: object, path: str | Path) -> None:
    """Serialise *data* to JSON at *path*, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> object:
    """Load JSON from *path*."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not already exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
