# Automated QA Tests

These tests were used during development to repeatedly check the extraction, matching, classification and export behaviour behind the final notebooks.

## What They Cover

- Recommendation and response-unit extraction for known hard documents, including Post Office, Behaviour Change, Grenfell, Covid and Summer 2024 examples.
- Structured response matching, including grouped labels, ordinal matching and document-specific numbering patterns.
- Classification regressions for accepted, partially accepted, rejected and not-addressed responses.
- Final export validation for `outputs/final_recommendations_246.json`.
- Frontend evaluation-summary utilities used by the optional local web app.

The tests are QA evidence for regression behaviour and known hard cases. They are not a substitute for a complete 246-row manual ground-truth benchmark.

## Running Locally

Run Python tests from the repository root after installing the default requirements:

```powershell
python -m pytest tests
```

Run the small JavaScript utility test with Node:

```powershell
node tests/evaluation-summary-utils.test.js
```
