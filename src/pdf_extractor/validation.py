"""Validation and confidence scoring."""
from __future__ import annotations

from typing import List

from .config import ValidationThresholds
from .models import Table, ValidationResult
from .utils import is_currency, is_date, is_number, parse_number


def infer_column_types(rows: List[List[str]]) -> List[str]:
    if not rows:
        return []
    n_cols = max(len(row) for row in rows)
    column_types: List[str] = []
    for col_idx in range(n_cols):
        values = [row[col_idx] for row in rows if col_idx < len(row) and row[col_idx]]
        if not values:
            column_types.append("text")
            continue
        date_hits = sum(1 for value in values if is_date(value))
        currency_hits = sum(1 for value in values if is_currency(value))
        number_hits = sum(1 for value in values if is_number(value))
        total = len(values)
        numeric_hits = number_hits + currency_hits
        if date_hits / total >= 0.6:
            column_types.append("date")
        elif numeric_hits / total >= 0.6:
            column_types.append("number")
        else:
            column_types.append("text")
    return column_types


def _type_integrity(rows: List[List[str]], column_types: List[str]) -> float:
    if not rows or not column_types:
        return 0.0
    total_checks = 0
    successes = 0
    for row in rows:
        for idx, col_type in enumerate(column_types):
            if idx >= len(row):
                continue
            value = row[idx]
            if not value:
                continue
            if col_type == "date":
                total_checks += 1
                successes += 1 if is_date(value) else 0
            elif col_type == "number":
                total_checks += 1
                successes += 1 if parse_number(value) is not None else 0
    if total_checks == 0:
        return 1.0
    return successes / total_checks


def validate_table(table: Table, thresholds: ValidationThresholds) -> ValidationResult:
    rows = table.rows()
    header_rows = set(table.header_rows)
    data_rows = [row for idx, row in enumerate(rows) if idx not in header_rows]

    total_cells = len(rows) * table.n_cols if rows else 0
    non_empty_cells = sum(1 for row in rows for value in row if value)
    coverage = non_empty_cells / total_cells if total_cells else 0.0

    non_empty_rows = sum(1 for row in rows if any(value for value in row))
    structural_consistency = non_empty_rows / max(1, len(rows))
    if rows and table.n_cols:
        col_fill_rates = []
        for col_idx in range(table.n_cols):
            filled = sum(1 for row in rows if col_idx < len(row) and row[col_idx])
            col_fill_rates.append(filled / max(1, len(rows)))
        if col_fill_rates:
            structural_consistency = min(structural_consistency, sum(col_fill_rates) / len(col_fill_rates))

    row_completeness = 0.0
    if data_rows:
        row_completeness = sum(
            (sum(1 for value in row if value) / max(1, table.n_cols)) for row in data_rows
        ) / len(data_rows)

    column_types = infer_column_types(data_rows)
    type_integrity = _type_integrity(data_rows, column_types)

    confidence = (
        coverage * 0.25
        + type_integrity * 0.35
        + row_completeness * 0.2
        + structural_consistency * 0.2
    )

    issues: List[str] = []
    if coverage < thresholds.min_coverage:
        issues.append("coverage_below_threshold")
    if type_integrity < thresholds.min_type_integrity:
        issues.append("type_integrity_below_threshold")
    if row_completeness < thresholds.min_row_completeness:
        issues.append("row_completeness_below_threshold")
    if structural_consistency < thresholds.min_structural_consistency:
        issues.append("structural_consistency_below_threshold")

    return ValidationResult(
        coverage=coverage,
        type_integrity=type_integrity,
        row_completeness=row_completeness,
        structural_consistency=structural_consistency,
        confidence=confidence,
        issues=issues,
        column_types=column_types,
    )
