"""PDFPlumber-based extraction strategy."""
from __future__ import annotations

from typing import Dict, List, Tuple

import pdfplumber

from ..config import ExtractorConfig, StrategyConfig
from ..models import Table, TableCell
from ..table_structure import (
    RowGroup,
    assign_cells_from_rows,
    group_words_by_row,
    infer_column_boundaries,
    merge_to_header_columns,
    merge_sparse_columns,
)
from ..utils import is_currency, is_number, safe_join


Word = Dict[str, object]


def _filter_words(words: List[Word], bbox: Tuple[float, float, float, float]) -> List[Word]:
    x0, top, x1, bottom = bbox
    filtered = []
    for word in words:
        if word["x1"] < x0 or word["x0"] > x1 or word["bottom"] < top or word["top"] > bottom:
            continue
        filtered.append(word)
    return filtered


def _infer_header_rows(rows: List[RowGroup]) -> List[int]:
    if not rows:
        return []

    all_words = [word for row in rows for word in row.words]
    sizes = [float(word.get("size", 0.0)) for word in all_words if word.get("size")]
    median_size = sorted(sizes)[len(sizes) // 2] if sizes else 0.0

    candidates: List[int] = []
    for idx, row in enumerate(rows[:4]):
        if len(row.words) < 3:
            continue
        text = safe_join([str(w["text"]) for w in row.words])
        alpha_chars = sum(ch.isalpha() for ch in text)
        total_chars = max(1, len(text))
        alpha_ratio = alpha_chars / total_chars
        numeric_ratio = sum(
            1
            for w in row.words
            if is_number(str(w["text"])) or is_currency(str(w["text"]))
        ) / max(1, len(row.words))
        avg_size = sum(float(w.get("size", 0.0)) for w in row.words) / max(1, len(row.words))

        if alpha_ratio >= 0.55 and numeric_ratio <= 0.4 and avg_size >= median_size * 0.9:
            candidates.append(idx)

    if not candidates:
        return []

    max_words = max(len(rows[idx].words) for idx in candidates)
    return [idx for idx in candidates if len(rows[idx].words) >= max_words * 0.7]


def _build_header_from_rows(
    rows: List[RowGroup],
    header_rows: List[int],
    col_boundaries: List[float],
) -> List[str]:
    if not rows or not header_rows or len(col_boundaries) < 2:
        return []
    col_centers = [
        (col_boundaries[idx] + col_boundaries[idx + 1]) / 2.0
        for idx in range(len(col_boundaries) - 1)
    ]
    header_map: Dict[int, List[Tuple[float, str]]] = {idx: [] for idx in range(len(col_centers))}
    for row_idx in header_rows:
        if row_idx >= len(rows):
            continue
        for word in rows[row_idx].words:
            x_center = (float(word["x0"]) + float(word["x1"])) / 2.0
            col_idx = min(range(len(col_centers)), key=lambda idx: abs(col_centers[idx] - x_center))
            header_map[col_idx].append((float(word["x0"]), str(word["text"])))

    headers: List[str] = []
    for col_idx in range(len(col_centers)):
        words_sorted = [text for _, text in sorted(header_map[col_idx], key=lambda x: x[0])]
        headers.append(safe_join(words_sorted))
    return headers


def _build_table(
    page_number: int,
    bbox: Tuple[float, float, float, float],
    words: List[Word],
    strategy: StrategyConfig,
) -> Table:
    row_groups = group_words_by_row(words, strategy.clustering.y_tolerance)
    header_rows = _infer_header_rows(row_groups)
    data_rows = [row for idx, row in enumerate(row_groups) if idx not in header_rows]
    if not data_rows:
        data_rows = row_groups
    col_boundaries, _ = infer_column_boundaries(
        data_rows,
        bbox,
        strategy.clustering.x_tolerance,
        strategy.clustering.gap_ratio,
        strategy.clustering.min_gap,
        strategy.clustering.boundary_support_ratio,
        strategy.clustering.boundary_min_support,
    )
    cells = assign_cells_from_rows(row_groups, col_boundaries)

    matrix = [["" for _ in range(max(1, len(col_boundaries) - 1))] for _ in range(len(row_groups))]
    for cell in cells:
        if cell.row_idx < len(matrix) and cell.col_idx < len(matrix[cell.row_idx]):
            matrix[cell.row_idx][cell.col_idx] = cell.text

    matrix, col_boundaries = merge_sparse_columns(
        matrix,
        col_boundaries,
        strategy.clustering.min_column_fill,
    )
    header_row_groups = [row_groups[idx] for idx in header_rows if idx < len(row_groups)]
    if header_row_groups:
        matrix, col_boundaries = merge_to_header_columns(
            matrix,
            header_row_groups,
            col_boundaries,
        )

    merged_cells = []
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            if value:
                merged_cells.append(TableCell(row_idx=row_idx, col_idx=col_idx, text=value))
    cells = merged_cells
    n_cols = max(1, len(col_boundaries) - 1)
    n_rows = max(1, len(row_groups))
    header_rows = _infer_header_rows(row_groups)
    header_labels = _build_header_from_rows(row_groups, header_rows, col_boundaries)

    return Table(
        page_number=page_number,
        bbox=bbox,
        cells=cells,
        n_rows=n_rows,
        n_cols=n_cols,
        header_rows=header_rows,
        strategy=strategy.name,
        metadata={"col_boundaries": col_boundaries, "header": header_labels},
    )


class PDFPlumberExtractor:
    def __init__(self, config: ExtractorConfig) -> None:
        self.config = config

    def extract_page_tables(self, page: pdfplumber.page.Page) -> List[Table]:
        words = page.extract_words(extra_attrs=["fontname", "size"], use_text_flow=True)
        tables: List[Table] = []

        for strategy in self.config.strategies:
            candidates = page.find_tables(strategy.table_settings)
            for candidate in candidates:
                bbox = tuple(candidate.bbox)  # type: ignore[arg-type]
                table_words = _filter_words(words, bbox)
                if len(table_words) < self.config.min_table_words:
                    continue
                table = _build_table(page.page_number, bbox, table_words, strategy)
                tables.append(table)

        if not tables and words:
            bbox = (
                min(word["x0"] for word in words),
                min(word["top"] for word in words),
                max(word["x1"] for word in words),
                max(word["bottom"] for word in words),
            )
            fallback_strategy = self.config.strategies[-1]
            table = _build_table(page.page_number, bbox, words, fallback_strategy)
            tables.append(table)

        return tables
