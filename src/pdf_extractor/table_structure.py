"""Grid inference and cell assignment for table extraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .models import TableCell, BBox
from .utils import safe_join


Word = Dict[str, object]


@dataclass
class RowGroup:
    words: List[Word]
    top: float
    bottom: float
    y_center: float


def _cluster_positions(values: List[float], tolerance: float) -> List[List[float]]:
    if not values:
        return []
    sorted_vals = sorted(values)
    clusters: List[List[float]] = [[sorted_vals[0]]]
    for value in sorted_vals[1:]:
        if abs(value - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return clusters


def _cluster_positions_with_counts(values: List[float], tolerance: float) -> List[Tuple[float, int]]:
    clusters = _cluster_positions(values, tolerance)
    return [(sum(cluster) / len(cluster), len(cluster)) for cluster in clusters]


def group_words_by_row(words: List[Word], y_tolerance: float) -> List[RowGroup]:
    if not words:
        return []
    words_sorted = sorted(
        words,
        key=lambda w: ((float(w["top"]) + float(w["bottom"])) / 2.0, float(w["x0"])),
    )
    rows: List[RowGroup] = []
    for word in words_sorted:
        y_center = (float(word["top"]) + float(word["bottom"])) / 2.0
        if not rows or abs(y_center - rows[-1].y_center) > y_tolerance:
            rows.append(RowGroup(words=[word], top=float(word["top"]), bottom=float(word["bottom"]), y_center=y_center))
        else:
            row = rows[-1]
            row.words.append(word)
            row.top = min(row.top, float(word["top"]))
            row.bottom = max(row.bottom, float(word["bottom"]))
            row.y_center = (row.y_center * (len(row.words) - 1) + y_center) / len(row.words)

    for row in rows:
        row.words.sort(key=lambda w: float(w["x0"]))

    return rows


def infer_column_boundaries(
    rows: List[RowGroup],
    table_bbox: BBox,
    x_tolerance: float,
    gap_ratio: float,
    min_gap: float,
    support_ratio: float,
    min_support: int,
) -> Tuple[List[float], List[Tuple[float, int]]]:
    x0, _, x1, _ = table_bbox
    gap_values: List[float] = []
    for row in rows:
        for left, right in zip(row.words, row.words[1:]):
            gap = float(right["x0"]) - float(left["x1"])
            if gap > 0:
                gap_values.append(gap)

    if gap_values:
        gap_values_sorted = sorted(gap_values)
        median_gap = gap_values_sorted[len(gap_values_sorted) // 2]
    else:
        median_gap = 0.0

    gap_threshold = max(min_gap, median_gap * gap_ratio)

    boundary_candidates: List[float] = []
    for row in rows:
        for left, right in zip(row.words, row.words[1:]):
            gap = float(right["x0"]) - float(left["x1"])
            if gap >= gap_threshold:
                boundary_candidates.append(float(left["x1"]) + gap / 2.0)

    clustered = _cluster_positions_with_counts(boundary_candidates, x_tolerance)
    min_required = max(min_support, int(len(rows) * support_ratio))
    boundaries = [x0]
    for center, count in clustered:
        if count >= min_required and x0 < center < x1:
            boundaries.append(center)
    boundaries.append(x1)
    boundaries = sorted(set(boundaries))

    return boundaries, clustered


def _find_index(center: float, boundaries: List[float]) -> int:
    for idx in range(len(boundaries) - 1):
        if boundaries[idx] <= center < boundaries[idx + 1]:
            return idx
    return max(0, len(boundaries) - 2)


def assign_cells_from_rows(rows: List[RowGroup], col_boundaries: List[float]) -> List[TableCell]:
    cell_map: Dict[Tuple[int, int], List[Tuple[float, str]]] = {}
    for row_idx, row in enumerate(rows):
        for word in row.words:
            x_center = (float(word["x0"]) + float(word["x1"])) / 2.0
            col_idx = _find_index(x_center, col_boundaries)
            cell_map.setdefault((row_idx, col_idx), []).append((float(word["x0"]), str(word["text"])))

    cells: List[TableCell] = []
    for (row_idx, col_idx), parts in cell_map.items():
        parts_sorted = [text for _, text in sorted(parts, key=lambda x: x[0])]
        cells.append(TableCell(row_idx=row_idx, col_idx=col_idx, text=safe_join(parts_sorted)))
    return cells


def merge_sparse_columns(
    rows: List[List[str]],
    col_boundaries: List[float],
    min_fill: float,
) -> Tuple[List[List[str]], List[float]]:
    if not rows or not col_boundaries:
        return rows, col_boundaries
    n_rows = len(rows)
    n_cols = len(rows[0]) if rows else 0

    def fill_rate(col_idx: int) -> float:
        filled = sum(1 for row in rows if col_idx < len(row) and row[col_idx])
        return filled / max(1, n_rows)

    col_idx = 0
    while col_idx < n_cols:
        rate = fill_rate(col_idx)
        if rate >= min_fill or n_cols <= 1:
            col_idx += 1
            continue
        merge_target = None
        if col_idx > 0:
            merge_target = col_idx - 1
        elif col_idx + 1 < n_cols:
            merge_target = col_idx + 1

        if merge_target is None:
            col_idx += 1
            continue

        for row in rows:
            left = row[merge_target]
            right = row[col_idx]
            row[merge_target] = safe_join([left, right]) if merge_target < col_idx else safe_join([right, left])
            del row[col_idx]
        if merge_target < col_idx:
            del col_boundaries[col_idx]
        else:
            del col_boundaries[col_idx + 1]
        n_cols -= 1
    return rows, col_boundaries


def merge_to_header_columns(
    rows: List[List[str]],
    header_rows: List[RowGroup],
    col_boundaries: List[float],
) -> Tuple[List[List[str]], List[float]]:
    if not header_rows or len(col_boundaries) <= 2:
        return rows, col_boundaries
    header_words = [word for row in header_rows for word in row.words]
    if not header_words:
        return rows, col_boundaries

    header_centers = sorted(
        (float(word["x0"]) + float(word["x1"])) / 2.0 for word in header_words
    )
    header_clusters = _cluster_positions_with_counts(header_centers, tolerance=8.0)
    if len(header_clusters) < 2:
        return rows, col_boundaries

    header_centers = [center for center, _ in header_clusters]
    target_cols = len(header_centers)
    current_cols = len(col_boundaries) - 1
    if current_cols <= target_cols:
        return rows, col_boundaries

    col_centers = [
        (col_boundaries[idx] + col_boundaries[idx + 1]) / 2.0
        for idx in range(current_cols)
    ]
    mapping = []
    for idx, center in enumerate(col_centers):
        target_idx = min(range(target_cols), key=lambda t: abs(header_centers[t] - center))
        mapping.append(target_idx)

    new_rows = []
    for row in rows:
        merged = ["" for _ in range(target_cols)]
        for col_idx, value in enumerate(row):
            if col_idx >= len(mapping):
                continue
            target_idx = mapping[col_idx]
            if merged[target_idx] and value:
                merged[target_idx] = safe_join([merged[target_idx], value])
            else:
                merged[target_idx] = merged[target_idx] or value
        new_rows.append(merged)

    boundaries = [col_boundaries[0]]
    for idx in range(1, target_cols):
        left = header_centers[idx - 1]
        right = header_centers[idx]
        boundaries.append((left + right) / 2.0)
    boundaries.append(col_boundaries[-1])
    boundaries = sorted(boundaries)
    return new_rows, boundaries


def infer_table_structure(
    words: List[Word],
    table_bbox: BBox,
    x_tol: float,
    y_tol: float,
    gap_ratio: float,
    min_gap: float,
    support_ratio: float,
    min_support: int,
    min_column_fill: float,
) -> Tuple[List[TableCell], List[RowGroup], List[float]]:
    rows = group_words_by_row(words, y_tol)
    col_boundaries, _ = infer_column_boundaries(
        rows,
        table_bbox,
        x_tol,
        gap_ratio,
        min_gap,
        support_ratio,
        min_support,
    )
    cells = assign_cells_from_rows(rows, col_boundaries)

    matrix = [["" for _ in range(max(1, len(col_boundaries) - 1))] for _ in range(len(rows))]
    for cell in cells:
        if cell.row_idx < len(matrix) and cell.col_idx < len(matrix[cell.row_idx]):
            matrix[cell.row_idx][cell.col_idx] = cell.text

    matrix, col_boundaries = merge_sparse_columns(matrix, col_boundaries, min_column_fill)

    merged_cells: List[TableCell] = []
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            if value:
                merged_cells.append(TableCell(row_idx=row_idx, col_idx=col_idx, text=value))

    return merged_cells, rows, col_boundaries
