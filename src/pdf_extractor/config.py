"""Configuration for extraction, validation, and continuation logic."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json


@dataclass
class ContinuationThresholds:
    max_col_boundary_drift_pts: float = 4.0
    max_col_boundary_drift_ratio: float = 0.02
    max_left_margin_shift_ratio: float = 0.02
    max_table_width_shift_ratio: float = 0.03
    header_strong_match: float = 0.80
    header_soft_match: float = 0.55
    alignment_match_ratio: float = 0.85


@dataclass
class ValidationThresholds:
    min_coverage: float = 0.99
    min_type_integrity: float = 0.95
    min_row_completeness: float = 0.99
    min_structural_consistency: float = 0.98
    min_confidence: float = 0.90


@dataclass
class ClusteringSettings:
    x_tolerance: float = 10.0
    y_tolerance: float = 4.0
    gap_ratio: float = 3.0
    min_gap: float = 8.0
    boundary_support_ratio: float = 0.3
    boundary_min_support: int = 4
    min_column_fill: float = 0.1


@dataclass
class StrategyConfig:
    name: str
    table_settings: Dict[str, Any]
    clustering: ClusteringSettings = field(default_factory=ClusteringSettings)


@dataclass
class OutputSettings:
    normalize_numbers: bool = True
    merge_same_header_groups: bool = True
    merged_output_prefix: str = "merged_group"


@dataclass
class OcrValidationSettings:
    enabled: bool = False
    engine: str = "tesseract"
    min_overlap: float = 0.6
    dpi: int = 200
    run_on_low_confidence: bool = True
    use_gpu: bool = False
    device: str = "mps"


@dataclass
class ProgressSettings:
    enabled: bool = True
    interval: int = 10


@dataclass
class ExtractorConfig:
    continuation: ContinuationThresholds = field(default_factory=ContinuationThresholds)
    validation: ValidationThresholds = field(default_factory=ValidationThresholds)
    strategies: List[StrategyConfig] = field(default_factory=list)
    output: OutputSettings = field(default_factory=OutputSettings)
    ocr_validation: OcrValidationSettings = field(default_factory=OcrValidationSettings)
    progress: ProgressSettings = field(default_factory=ProgressSettings)
    max_pages: Optional[int] = None
    batch_size: Optional[int] = 50
    router_pages: int = 3
    min_table_words: int = 10

    @staticmethod
    def default() -> "ExtractorConfig":
        strategies = [
            StrategyConfig(
                name="lattice",
                table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "intersection_tolerance": 5,
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                    "edge_min_length": 3,
                    "min_words_vertical": 3,
                    "min_words_horizontal": 1,
                    "text_x_tolerance": 2,
                    "text_y_tolerance": 2,
                },
                clustering=ClusteringSettings(
                    x_tolerance=8.0,
                    y_tolerance=3.0,
                    gap_ratio=3.0,
                    min_gap=8.0,
                    boundary_support_ratio=0.3,
                    boundary_min_support=4,
                    min_column_fill=0.08,
                ),
            ),
            StrategyConfig(
                name="stream",
                table_settings={
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                    "edge_min_length": 3,
                    "min_words_vertical": 3,
                    "min_words_horizontal": 1,
                    "text_x_tolerance": 3,
                    "text_y_tolerance": 3,
                },
                clustering=ClusteringSettings(
                    x_tolerance=12.0,
                    y_tolerance=5.0,
                    gap_ratio=3.0,
                    min_gap=8.0,
                    boundary_support_ratio=0.3,
                    boundary_min_support=4,
                    min_column_fill=0.08,
                ),
            ),
            StrategyConfig(
                name="stream_conservative",
                table_settings={
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                    "snap_tolerance": 4,
                    "join_tolerance": 4,
                    "edge_min_length": 3,
                    "min_words_vertical": 3,
                    "min_words_horizontal": 1,
                    "text_x_tolerance": 3,
                    "text_y_tolerance": 3,
                },
                clustering=ClusteringSettings(
                    x_tolerance=14.0,
                    y_tolerance=6.0,
                    gap_ratio=4.0,
                    min_gap=10.0,
                    boundary_support_ratio=0.3,
                    boundary_min_support=4,
                    min_column_fill=0.08,
                ),
            ),
        ]
        return ExtractorConfig(strategies=strategies)


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Optional[str]) -> ExtractorConfig:
    if not path:
        return ExtractorConfig.default()
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    defaults = ExtractorConfig.default()
    merged = _merge_dict(
        {
            "continuation": defaults.continuation.__dict__,
            "validation": defaults.validation.__dict__,
            "strategies": [
                {
                    "name": s.name,
                    "table_settings": s.table_settings,
                    "clustering": s.clustering.__dict__,
                }
                for s in defaults.strategies
            ],
            "output": defaults.output.__dict__,
            "ocr_validation": defaults.ocr_validation.__dict__,
            "progress": defaults.progress.__dict__,
            "max_pages": defaults.max_pages,
            "batch_size": defaults.batch_size,
            "router_pages": defaults.router_pages,
            "min_table_words": defaults.min_table_words,
        },
        raw,
    )

    strategies = []
    for strategy in merged.get("strategies", []):
        clustering = ClusteringSettings(**strategy.get("clustering", {}))
        strategies.append(
            StrategyConfig(
                name=strategy["name"],
                table_settings=strategy.get("table_settings", {}),
                clustering=clustering,
            )
        )

    return ExtractorConfig(
        continuation=ContinuationThresholds(**merged.get("continuation", {})),
        validation=ValidationThresholds(**merged.get("validation", {})),
        strategies=strategies,
        output=OutputSettings(**merged.get("output", {})),
        ocr_validation=OcrValidationSettings(**merged.get("ocr_validation", {})),
        progress=ProgressSettings(**merged.get("progress", {})),
        max_pages=merged.get("max_pages"),
        batch_size=merged.get("batch_size"),
        router_pages=merged.get("router_pages", defaults.router_pages),
        min_table_words=merged.get("min_table_words", defaults.min_table_words),
    )
