"""Post-processing utilities for merging split table outputs."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .utils import normalize_text


def _group_key(header: List[str]) -> Optional[Tuple[str, ...]]:
    if not header:
        return None
    normalized = tuple(normalize_text(cell) for cell in header)
    if not any(normalized):
        return None
    return normalized


def _extract_group_id(path: Path) -> int:
    name = path.stem
    parts = name.split("_")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return 0


def merge_groups_by_header(output_dir: str, prefix: str) -> List[Path]:
    output_path = Path(output_dir)
    csv_files = sorted(output_path.glob("table_group_*.csv"), key=_extract_group_id)
    groups: Dict[Tuple[str, ...], Dict[str, object]] = {}

    for path in csv_files:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                continue
        key = _group_key(header)
        if key is None:
            continue
        if key not in groups:
            groups[key] = {"header": header, "paths": []}
        groups[key]["paths"].append(path)

    merged_files: List[Path] = []
    for idx, (key, payload) in enumerate(groups.items(), start=1):
        paths = payload["paths"]
        header = payload["header"]
        if len(paths) <= 1:
            continue
        merged_path = output_path / f"{prefix}_{idx}.csv"
        with merged_path.open("w", newline="", encoding="utf-8") as out_handle:
            writer = csv.writer(out_handle)
            writer.writerow(header)
            for path in paths:
                with path.open("r", newline="", encoding="utf-8") as handle:
                    reader = csv.reader(handle)
                    try:
                        next(reader)
                    except StopIteration:
                        continue
                    for row in reader:
                        writer.writerow(row)
        merged_files.append(merged_path)

    return merged_files
