"""Convenience runner for local development without editable install."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pdf_extractor.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
