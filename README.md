PDF Extractor
=============

Dynamic PDF-to-CSV table extractor with document-class routing, validation, and
multi-page table continuation support. Designed for large vector PDFs with
consistent or semi-consistent layouts.

Features
- Vector-first extraction with dynamic grid inference
- Document-class routing (ledger, income statement, balance sheet, cash flow)
- Multi-page continuation detection and row stitching
- Validation and confidence scoring
- Automated reprocessing across strategies

Quick Start
- Create and activate venv
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`
- Install dependencies
  - `pip install -e .`

Run
- `PYTHONPATH=src python run.py --input "Cash Earnings.pdf" --output outputs`
- If editable install is available: `pdf-extract --input "Cash Earnings.pdf" --output outputs`

Config
- Defaults are built-in. Optionally override with JSON:
  - `pdf-extract --input "Cash Earnings.pdf" --output outputs --config configs/default.json`

Optional OCR Validation
- Install OCR extras: `pip install -e ".[ocr]"`
- Requires system Tesseract in PATH.
- GPU OCR requires an engine that supports it (e.g., PaddleOCR or DocTR) and a compatible install.
- For M1/M2, set `"engine": "doctr"` and `"use_gpu": true` to target Metal (MPS) when available.

Outputs
- CSV files per logical table group
- Validation report JSON in `reports/`
- Optional merged CSV outputs when groups share identical headers

Notes
- OCR and ML structure detection are not enabled by default. The pipeline is
  designed so these can be added as pluggable strategies later.
# PDF_Extractor_v2
