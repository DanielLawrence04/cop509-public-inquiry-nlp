"""Precompute the default hosted search corpus and save it to disk.

Run this script before committing/deploying so the backend can load the
corpus instantly on startup without re-processing PDFs during the Render
health-check window.

Usage (from the repo root):
    python scripts/precompute_search_corpus.py

The script processes all 8 default preset document pairs (both
coursework_given and extra_found), extracts and chunks pages, extracts
response units, and writes one pickle file per preset plus a manifest to:

    backend/data/prebuilt_search_corpus/

Commit the generated files.  On deploy, the FastAPI lifespan will load them
in milliseconds instead of rebuilding from PDFs.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project root is on sys.path so backend.* and src.* imports work.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "backend" / "data" / "prebuilt_search_corpus"
MANIFEST_PATH = OUT_DIR / "manifest.json"
SCHEMA_VERSION = 1


def main() -> None:
    from backend.core.presets import PRESETS, validate_preset_files
    from src.pdf_loader import extract_pages
    from src.chunking import chunk_pages_v2
    from src.response_units import extract_response_units

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_presets: dict = {}
    total_pairs = 0
    total_chunks = 0
    skipped: list[str] = []
    t_start = time.monotonic()

    for pid, preset in PRESETS.items():
        print(f"\n[precompute] {pid}...", flush=True)
        t0 = time.monotonic()

        try:
            validate_preset_files(preset)
        except FileNotFoundError as exc:
            print(f"  SKIP — PDF missing: {exc}", flush=True)
            skipped.append(pid)
            continue

        print(f"  extracting pages from {preset.policy_pdf.name}...", flush=True)
        policy_pages = list(extract_pages(preset.policy_pdf))
        print(f"  extracting pages from {preset.response_pdf.name}...", flush=True)
        response_pages = list(extract_pages(preset.response_pdf))

        print(f"  chunking {len(policy_pages)} policy pages...", flush=True)
        policy_chunks = chunk_pages_v2(policy_pages)
        print(f"  chunking {len(response_pages)} response pages...", flush=True)
        response_chunks = chunk_pages_v2(response_pages)

        print("  extracting response units...", flush=True)
        try:
            response_units = extract_response_units(response_pages)
        except Exception as exc:
            print(f"  WARNING: response_units failed ({exc}), storing empty list", flush=True)
            response_units = []

        load_ocr = any(p.get("ocr") for p in policy_pages) or any(
            p.get("ocr") for p in response_pages
        )

        data = {
            "schema_version": SCHEMA_VERSION,
            "preset_id": pid,
            "policy_chunks": policy_chunks,
            "response_chunks": response_chunks,
            "response_units": response_units,
            "policy_pages": policy_pages,
            "response_pages": response_pages,
            "load_ocr": load_ocr,
        }

        out_path = OUT_DIR / f"{pid}.pkl"
        with open(out_path, "wb") as fh:
            pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        n_policy = len(policy_chunks)
        n_response = len(response_chunks)
        n_units = len(response_units)
        manifest_presets[pid] = {
            "policy_chunks": n_policy,
            "response_chunks": n_response,
            "response_units": n_units,
        }
        total_pairs += 1
        total_chunks += n_policy + n_response
        print(
            f"  done: {n_policy} policy + {n_response} response chunks, "
            f"{n_units} units ({elapsed_ms} ms)",
            flush=True,
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "presets": manifest_presets,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total_ms = int((time.monotonic() - t_start) * 1000)
    print(
        f"\n[precompute] done: {total_pairs} pairs, {total_chunks} chunks "
        f"in {total_ms} ms",
        flush=True,
    )
    if skipped:
        print(f"[precompute] skipped (PDFs missing): {', '.join(skipped)}", flush=True)
    print(f"[precompute] corpus saved to: {OUT_DIR}", flush=True)
    print(
        "[precompute] commit the generated .pkl and manifest.json files "
        "so Render can load them on deploy.",
        flush=True,
    )


if __name__ == "__main__":
    main()
