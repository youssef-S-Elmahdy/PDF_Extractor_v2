"""Continuation logic for multi-page tables."""
from __future__ import annotations

from typing import List, Tuple

from .config import ContinuationThresholds
from .models import Table, TableSignature
from .utils import jaccard_similarity, tokenize_header
from .validation import infer_column_types


def _header_tokens(table: Table) -> set[str]:
    header_rows = table.header_rows
    if not header_rows:
        return set()
    rows = table.rows()
    tokens: List[str] = []
    for row_idx in header_rows:
        if row_idx < len(rows):
            for cell in rows[row_idx]:
                tokens.extend(tokenize_header(cell))
    return set(tokens)


def _alignment_pattern(rows: List[List[str]]) -> List[str]:
    column_types = infer_column_types(rows)
    pattern = []
    for col_type in column_types:
        if col_type in {"number", "currency"}:
            pattern.append("R")
        elif col_type == "date":
            pattern.append("C")
        else:
            pattern.append("L")
    return pattern


def build_signature(table: Table) -> TableSignature:
    col_boundaries = table.metadata.get("col_boundaries")
    if not col_boundaries:
        x0, _, x1, _ = table.bbox
        width = (x1 - x0) / max(1, table.n_cols)
        col_boundaries = [x0 + width * idx for idx in range(table.n_cols + 1)]
    col_widths = [col_boundaries[i + 1] - col_boundaries[i] for i in range(len(col_boundaries) - 1)]
    header_tokens = _header_tokens(table)
    rows = table.rows()
    pattern = _alignment_pattern(rows)
    x0, top, x1, bottom = table.bbox

    return TableSignature(
        n_cols=table.n_cols,
        col_boundaries=col_boundaries,
        col_widths=col_widths,
        header_tokens=header_tokens,
        alignment_pattern=pattern,
        table_left=x0,
        table_right=x1,
        table_top=top,
        table_bottom=bottom,
    )


def _boundary_drift(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        return float("inf")
    return sum(abs(a[i] - b[i]) for i in range(len(a))) / len(a)


def _alignment_match(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    size = min(len(a), len(b))
    matches = sum(1 for idx in range(size) if a[idx] == b[idx])
    return matches / size


def compare_signatures(
    prev: TableSignature,
    curr: TableSignature,
    thresholds: ContinuationThresholds,
    page_width: float,
) -> Tuple[bool, float]:
    if prev.n_cols != curr.n_cols:
        return False, 0.0

    boundary_drift = _boundary_drift(prev.col_boundaries, curr.col_boundaries)
    drift_ratio = boundary_drift / max(1.0, page_width)
    if boundary_drift > thresholds.max_col_boundary_drift_pts and drift_ratio > thresholds.max_col_boundary_drift_ratio:
        return False, 0.0

    left_shift = abs(prev.table_left - curr.table_left) / max(1.0, page_width)
    width_shift = abs((prev.table_right - prev.table_left) - (curr.table_right - curr.table_left)) / max(1.0, page_width)
    if left_shift > thresholds.max_left_margin_shift_ratio or width_shift > thresholds.max_table_width_shift_ratio:
        return False, 0.0

    header_similarity = jaccard_similarity(prev.header_tokens, curr.header_tokens)
    alignment_similarity = _alignment_match(prev.alignment_pattern, curr.alignment_pattern)

    geometry_score = 1.0 - min(1.0, boundary_drift / max(1.0, thresholds.max_col_boundary_drift_pts))
    placement_score = 1.0 - min(1.0, (left_shift + width_shift))

    score = (
        geometry_score * 0.45
        + header_similarity * 0.25
        + alignment_similarity * 0.15
        + placement_score * 0.15
    )

    if curr.header_tokens and header_similarity < thresholds.header_soft_match:
        return False, score
    if alignment_similarity < thresholds.alignment_match_ratio * 0.8:
        return False, score

    return score >= 0.75, score
