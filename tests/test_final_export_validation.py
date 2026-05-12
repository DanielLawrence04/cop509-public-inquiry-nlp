from __future__ import annotations

from functools import lru_cache

from scripts.validate_recommendation_export import (
    classification_failures,
    generate_export_rows,
    metadata_failures,
    recommendation_cleanup_failures,
    response_leakage_failures,
)


@lru_cache(maxsize=1)
def _rows():
    return tuple(generate_export_rows())


def test_export_metadata_complete():
    assert not metadata_failures(list(_rows()))


def test_export_response_leakage_cleaned():
    rows = list(_rows())
    failures = response_leakage_failures(rows) + recommendation_cleanup_failures(rows)
    assert not failures


def test_export_classification_corrections():
    assert not classification_failures(list(_rows()))
