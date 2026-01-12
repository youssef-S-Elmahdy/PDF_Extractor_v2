"""Extraction pipeline orchestrating routing, validation, and output."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import gc

import pdfplumber
from pypdf import PdfReader

from .config import ExtractorConfig
from .continuation import build_signature, compare_signatures
from .extractors.pdfplumber_extractor import PDFPlumberExtractor
from .models import Table, TableGroup
from .ocr_validation import OcrVerifier
from .output import OutputWriter
from .postprocess import merge_groups_by_header
from .router import DocumentRouter
from .utils import jaccard_similarity, normalize_numeric, tokenize_header
from .validation import ValidationResult, validate_table


@dataclass
class GroupState:
    group: TableGroup
    last_page: int
    pending_row: Optional[List[str]] = None
    column_types: List[str] = field(default_factory=list)


def _bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax0, atop, ax1, abot = a
    bx0, btop, bx1, bbot = b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(atop, btop)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(abot, bbot)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    area_a = (ax1 - ax0) * (abot - atop)
    area_b = (bx1 - bx0) * (bbot - btop)
    union = area_a + area_b - inter_area
    return inter_area / union if union else 0.0


def _group_candidates(candidates: List[Table]) -> List[List[Table]]:
    groups: List[List[Table]] = []
    for candidate in candidates:
        matched = False
        for group in groups:
            if _bbox_iou(group[0].bbox, candidate.bbox) >= 0.7:
                group.append(candidate)
                matched = True
                break
        if not matched:
            groups.append([candidate])
    return groups


def _best_candidate(group: List[Table]) -> Table:
    return max(group, key=lambda table: table.confidence)


def _header_from_table(table: Table) -> List[str]:
    header_override = table.metadata.get("header")
    if header_override:
        return header_override
    rows = table.rows()
    if not table.header_rows:
        return []
    headers = ["" for _ in range(table.n_cols)]
    for row_idx in table.header_rows:
        if row_idx >= len(rows):
            continue
        for col_idx, value in enumerate(rows[row_idx]):
            if not value:
                continue
            if headers[col_idx]:
                headers[col_idx] = f"{headers[col_idx]} {value}".strip()
            else:
                headers[col_idx] = value
    return headers


def _rows_without_headers(table: Table) -> List[List[str]]:
    rows = table.rows()
    header_rows = set(table.header_rows)
    return [row for idx, row in enumerate(rows) if idx not in header_rows]


def _is_header_repeat(row: List[str], header: List[str]) -> bool:
    if not header or not row:
        return False
    row_tokens = []
    for cell in row:
        row_tokens.extend(tokenize_header(cell))
    header_tokens = []
    for cell in header:
        header_tokens.extend(tokenize_header(cell))
    return jaccard_similarity(row_tokens, header_tokens) >= 0.8


def _should_merge(pending: List[str], current: List[str], column_types: List[str]) -> bool:
    if not pending or not current:
        return False
    numeric_cols = [idx for idx, col_type in enumerate(column_types) if col_type in {"number", "currency"}]
    if not numeric_cols:
        return False
    pending_numeric = sum(1 for idx in numeric_cols if idx < len(pending) and pending[idx])
    current_numeric = sum(1 for idx in numeric_cols if idx < len(current) and current[idx])
    pending_text = sum(1 for value in pending if value)
    current_text = sum(1 for value in current if value)
    return pending_numeric == 0 and current_numeric == 0 and pending_text > 0 and current_text > 0


def _merge_rows(pending: List[str], current: List[str]) -> List[str]:
    merged = []
    width = max(len(pending), len(current))
    for idx in range(width):
        left = pending[idx] if idx < len(pending) else ""
        right = current[idx] if idx < len(current) else ""
        if left and right:
            merged.append(f"{left} {right}".strip())
        else:
            merged.append(left or right)
    return merged


def _normalize_row(row: List[str], column_types: List[str]) -> List[str]:
    if not column_types:
        return row
    normalized = list(row)
    for idx, col_type in enumerate(column_types):
        if idx >= len(normalized):
            break
        if col_type == "number":
            normalized[idx] = normalize_numeric(normalized[idx])
    return normalized


class ExtractorPipeline:
    def __init__(self, config: ExtractorConfig) -> None:
        self.config = config
        self.router = DocumentRouter()
        self.extractor = PDFPlumberExtractor(config)
        self.ocr_verifier = OcrVerifier(config.ocr_validation)

    def _normalize_output_row(self, row: List[str], column_types: List[str]) -> List[str]:
        if not self.config.output.normalize_numbers:
            return row
        return _normalize_row(row, column_types)

    def _apply_ocr_validation(self, page: pdfplumber.page.Page, table: Table) -> None:
        if not self.ocr_verifier.available:
            return
        if not self.ocr_verifier.should_verify(table, self.config.validation.min_confidence):
            return
        engine = self.ocr_verifier.engine_name
        device = self.ocr_verifier.device
        device_label = f" ({device})" if device else ""
        if self.config.progress.enabled:
            print(
                f"[ocr] start page {page.page_number} engine={engine}{device_label}",
                flush=True,
            )
        result = self.ocr_verifier.verify(page, table)
        if not result:
            return
        table.metadata["ocr_overlap"] = result.overlap
        table.metadata["ocr_token_count"] = result.token_count
        table.metadata["ocr_engine"] = result.engine
        if result.device:
            table.metadata["ocr_device"] = result.device
        table.confidence = self.ocr_verifier.combine_confidence(table.confidence, result.overlap)
        if self.config.progress.enabled:
            print(
                f"[ocr] done page {page.page_number} overlap={result.overlap:.2f}",
                flush=True,
            )

    def run(self, pdf_path: str, output_dir: str, report_dir: str) -> None:
        output = OutputWriter(output_dir, report_dir)
        group_states: List[GroupState] = []
        group_id = 1
        total_rows = 0
        interval_rows = 0

        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        if self.config.max_pages:
            total_pages = min(total_pages, self.config.max_pages)

        with pdfplumber.open(pdf_path) as pdf:
            page_texts = []
            for page in pdf.pages[: min(self.config.router_pages, total_pages)]:
                page_texts.append(page.extract_text() or "")
            route = self.router.classify(page_texts)

        batch_size = self.config.batch_size or total_pages
        for start in range(0, total_pages, batch_size):
            end = min(start + batch_size, total_pages)
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages[start:end]:
                    if self.config.max_pages and page.page_number > self.config.max_pages:
                        break
                    candidates = self.extractor.extract_page_tables(page)
                    for candidate in candidates:
                        validation = validate_table(candidate, self.config.validation)
                        candidate.confidence = validation.confidence
                        candidate.metadata["validation"] = validation

                    grouped = _group_candidates(candidates)
                    selected: List[Table] = []
                    for group in grouped:
                        ordered = sorted(group, key=lambda table: table.confidence, reverse=True)
                        best = ordered[0]
                        self._apply_ocr_validation(page, best)
                        best_overlap = best.metadata.get("ocr_overlap")

                        if (
                            self.ocr_verifier.available
                            and best_overlap is not None
                            and best_overlap < self.config.ocr_validation.min_overlap
                        ):
                            for alt in ordered[1:]:
                                self._apply_ocr_validation(page, alt)
                                alt_overlap = alt.metadata.get("ocr_overlap")
                                if alt_overlap is None:
                                    continue
                                if alt_overlap >= best_overlap:
                                    best = alt
                                    best_overlap = alt_overlap
                                    if best_overlap >= self.config.ocr_validation.min_overlap:
                                        break
                        selected.append(best)

                    page_rows_log = 0
                    for table in selected:
                        validation = table.metadata["validation"]
                        signature = build_signature(table)
                        matched_state = None
                        matched_score = 0.0
                        for state in group_states:
                            if page.page_number - state.last_page > 1:
                                continue
                            matches, score = compare_signatures(
                                state.group.signature,
                                signature,
                                self.config.continuation,
                                page.width,
                            )
                            if matches and score > matched_score:
                                matched_state = state
                                matched_score = score

                        if matched_state is None:
                            header = _header_from_table(table)
                            group = TableGroup(
                                group_id=group_id,
                                class_name=route.class_name,
                                signature=signature,
                                header=header,
                            )
                            group_id += 1
                            state = GroupState(group=group, last_page=page.page_number)
                            output.open_group(group)
                            group_states.append(state)
                            matched_state = state
                        else:
                            matched_state.last_page = page.page_number
                            matched_state.group.signature = signature

                        if not matched_state.group.header:
                            matched_state.group.header = _header_from_table(table)

                        if not matched_state.column_types:
                            matched_state.column_types = validation.column_types

                        rows = _rows_without_headers(table)
                        if rows and _is_header_repeat(rows[0], matched_state.group.header):
                            rows = rows[1:]
                        page_rows_log += len(rows)

                        for row in rows:
                            if matched_state.pending_row is None:
                                matched_state.pending_row = row
                                continue
                            if _should_merge(matched_state.pending_row, row, matched_state.column_types):
                                matched_state.pending_row = _merge_rows(matched_state.pending_row, row)
                            else:
                                output.write_rows(
                                    matched_state.group,
                                    [
                                        self._normalize_output_row(
                                            matched_state.pending_row,
                                            matched_state.column_types,
                                        )
                                    ],
                                )
                                total_rows += 1
                                matched_state.pending_row = row

                        extra = {}
                        if "ocr_overlap" in table.metadata:
                            extra["ocr_overlap"] = table.metadata["ocr_overlap"]
                            extra["ocr_token_count"] = table.metadata.get("ocr_token_count")
                            extra["ocr_engine"] = table.metadata.get("ocr_engine")
                            extra["ocr_device"] = table.metadata.get("ocr_device")
                        output.add_report(
                            matched_state.group.group_id,
                            page.page_number,
                            validation,
                            extra=extra or None,
                        )

                    if self.config.progress.enabled:
                        interval = max(1, self.config.progress.interval)
                        interval_rows += page_rows_log
                        if page.page_number % interval == 0 or page.page_number == total_pages:
                            label = f"page {page.page_number}"
                            print(
                                f"[progress] {label}/{total_pages}, extracted {interval_rows} rows",
                                flush=True,
                            )
                            interval_rows = 0

            gc.collect()

        for state in group_states:
            if state.pending_row is not None:
                output.write_rows(
                    state.group,
                    [self._normalize_output_row(state.pending_row, state.column_types)],
                )
                total_rows += 1
                state.pending_row = None

        output.write_report()
        output.close()
        if self.config.progress.enabled:
            print(f"[done] extracted total {total_rows} rows", flush=True)
        if self.config.output.merge_same_header_groups:
            merged = merge_groups_by_header(output_dir, self.config.output.merged_output_prefix)
            if self.config.progress.enabled and merged:
                print(
                    f"[merge] merged {len(merged)} csv outputs",
                    flush=True,
                )
