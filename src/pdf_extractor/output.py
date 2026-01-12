"""Output writers for CSV and validation reports."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from .models import TableGroup, ValidationResult


class OutputWriter:
    def __init__(self, output_dir: str, report_dir: str) -> None:
        self.output_path = Path(output_dir)
        self.report_path = Path(report_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.report_path.mkdir(parents=True, exist_ok=True)
        self._writers: Dict[int, csv.writer] = {}
        self._files: Dict[int, any] = {}
        self._reports: Dict[int, List[dict]] = {}
        self._header_written: Dict[int, bool] = {}
        self._column_maps: Dict[int, Optional[List[int]]] = {}

    def open_group(self, group: TableGroup) -> None:
        if group.group_id in self._header_written:
            return
        self._reports[group.group_id] = []
        self._header_written[group.group_id] = False
        self._column_maps[group.group_id] = None

    def _ensure_writer(self, group: TableGroup) -> None:
        if group.group_id in self._writers:
            return
        filename = f"table_group_{group.group_id}.csv"
        file_path = self.output_path / filename
        handle = open(file_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(handle)
        self._writers[group.group_id] = writer
        self._files[group.group_id] = handle

    @staticmethod
    def _row_has_data(row: List[str]) -> bool:
        return any(cell for cell in row)

    @staticmethod
    def _collapse_header(header: List[str]) -> tuple[List[str], List[int]]:
        if not header:
            return header, []
        non_empty_indices = [idx for idx, value in enumerate(header) if value.strip()]
        if not non_empty_indices:
            return header, list(range(len(header)))
        collapsed = [header[idx] for idx in non_empty_indices]
        index_map = {idx: pos for pos, idx in enumerate(non_empty_indices)}
        mapping: List[int] = []
        for idx, value in enumerate(header):
            if value.strip():
                mapping.append(index_map[idx])
                continue
            left = next((i for i in reversed(non_empty_indices) if i < idx), None)
            right = next((i for i in non_empty_indices if i > idx), None)
            if left is not None:
                mapping.append(index_map[left])
            elif right is not None:
                mapping.append(index_map[right])
            else:
                mapping.append(0)
        return collapsed, mapping

    @staticmethod
    def _apply_mapping(row: List[str], mapping: List[int], target_len: int) -> List[str]:
        if not mapping or target_len <= 0:
            return row
        merged = ["" for _ in range(target_len)]
        for idx, value in enumerate(row):
            if idx >= len(mapping):
                continue
            target = mapping[idx]
            if not value:
                continue
            if merged[target]:
                merged[target] = f"{merged[target]} {value}".strip()
            else:
                merged[target] = value
        return merged

    def write_rows(self, group: TableGroup, rows: List[List[str]]) -> None:
        column_map = self._column_maps.get(group.group_id)
        if column_map is None and group.header:
            collapsed_header, mapping = self._collapse_header(group.header)
            if collapsed_header != group.header:
                group.header = collapsed_header
                column_map = mapping
            else:
                column_map = mapping
            self._column_maps[group.group_id] = column_map

        if column_map:
            rows = [self._apply_mapping(row, column_map, len(group.header)) for row in rows]

        if not any(self._row_has_data(row) for row in rows):
            return
        self._ensure_writer(group)
        writer = self._writers[group.group_id]
        header_written = self._header_written.get(group.group_id, False)
        if not header_written and group.header:
            writer.writerow(group.header)
            self._header_written[group.group_id] = True
        for row in rows:
            if not self._row_has_data(row):
                continue
            writer.writerow(row)
            group.rows_written += 1

    def add_report(
        self,
        group_id: int,
        page_number: int,
        validation: ValidationResult,
        extra: Optional[dict] = None,
    ) -> None:
        payload = {
            "page_number": page_number,
            "validation": asdict(validation),
        }
        if extra:
            payload.update(extra)
        self._reports[group_id].append(payload)

    def close(self) -> None:
        for handle in self._files.values():
            handle.close()
        self._files.clear()

    def write_report(self) -> None:
        report_file = self.report_path / "validation_report.json"
        with open(report_file, "w", encoding="utf-8") as handle:
            json.dump(self._reports, handle, indent=2)
