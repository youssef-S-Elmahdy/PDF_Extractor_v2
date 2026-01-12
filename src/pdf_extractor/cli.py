"""Command-line interface for PDF extraction."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .pipeline import ExtractorPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dynamic PDF-to-CSV table extractor")
    parser.add_argument("--input", required=True, help="Path to input PDF")
    parser.add_argument("--output", default="outputs", help="Output directory for CSVs")
    parser.add_argument("--report", default="reports", help="Output directory for reports")
    parser.add_argument("--config", default=None, help="Path to config JSON")
    parser.add_argument("--max-pages", type=int, default=None, help="Limit number of pages")
    parser.add_argument("--batch-size", type=int, default=None, help="Process pages in batches")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input PDF not found: {input_path}")

    config = load_config(args.config)
    if args.max_pages:
        config.max_pages = args.max_pages
    if args.batch_size:
        config.batch_size = args.batch_size

    pipeline = ExtractorPipeline(config)
    pipeline.run(str(input_path), args.output, args.report)


if __name__ == "__main__":
    main()
