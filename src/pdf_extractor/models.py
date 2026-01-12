"""Data models for tables and validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

BBox = Tuple[float, float, float, float]


@dataclass
class TableCell:
    row_idx: int
    col_idx: int
    text: str
    bbox: Optional[BBox] = None


@dataclass
class Table:
    page_number: int
    bbox: BBox
    cells: List[TableCell]
    n_rows: int
    n_cols: int
    header_rows: List[int]
    strategy: str
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def rows(self) -> List[List[str]]:
        matrix = [["" for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        for cell in self.cells:
            if 0 <= cell.row_idx < self.n_rows and 0 <= cell.col_idx < self.n_cols:
                matrix[cell.row_idx][cell.col_idx] = cell.text
        return matrix


@dataclass
class TableSignature:
    n_cols: int
    col_boundaries: List[float]
    col_widths: List[float]
    header_tokens: set[str]
    alignment_pattern: List[str]
    table_left: float
    table_right: float
    table_top: float
    table_bottom: float


@dataclass
class TableGroup:
    group_id: int
    class_name: str
    signature: TableSignature
    header: List[str]
    rows_written: int = 0
    tables: List[Table] = field(default_factory=list)


@dataclass
class ValidationResult:
    coverage: float
    type_integrity: float
    row_completeness: float
    structural_consistency: float
    confidence: float
    issues: List[str] = field(default_factory=list)
    column_types: List[str] = field(default_factory=list)
