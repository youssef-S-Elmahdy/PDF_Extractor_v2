"""OCR-based secondary validation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pdfplumber

from .config import OcrValidationSettings
from .models import Table
from .utils import tokenize_text


@dataclass
class OcrResult:
    overlap: float
    token_count: int
    engine: str
    device: Optional[str] = None


class OcrVerifier:
    def __init__(self, settings: OcrValidationSettings) -> None:
        self.settings = settings
        self.engine_name = settings.engine
        self.device: Optional[str] = None
        self.engine = None
        self.available = False
        self._init_engine()

    def _init_engine(self) -> None:
        if not self.settings.enabled:
            return
        engine = self.settings.engine.lower()
        if engine == "tesseract":
            try:
                import pytesseract  # noqa: F401
            except ImportError:
                return
            self.engine = "tesseract"
            self.available = True
            return

        if engine == "paddle":
            try:
                from paddleocr import PaddleOCR
            except ImportError:
                return
            self.engine = PaddleOCR(use_gpu=self.settings.use_gpu, show_log=False)
            self.available = True
            return

        if engine == "doctr":
            try:
                import torch
                from doctr.models import ocr_predictor
            except ImportError:
                return
            device = "cpu"
            if self.settings.use_gpu:
                if self.settings.device and self.settings.device != "cpu":
                    device = self.settings.device
                elif torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
            predictor = ocr_predictor(pretrained=True)
            if hasattr(predictor, "to"):
                predictor = predictor.to(device)
            self.engine = predictor
            self.device = device
            self.available = True

    def should_verify(self, table: Table, min_confidence: float) -> bool:
        if not self.available:
            return False
        if not self.settings.run_on_low_confidence:
            return True
        validation = table.metadata.get("validation")
        has_issues = bool(getattr(validation, "issues", []))
        return table.confidence < min_confidence or has_issues

    def verify(self, page: pdfplumber.page.Page, table: Table) -> Optional[OcrResult]:
        if not self.available:
            return None

        cropped = page.crop(table.bbox)
        image = cropped.to_image(resolution=self.settings.dpi).original
        engine = self.settings.engine.lower()
        ocr_text = ""

        if engine == "tesseract":
            try:
                import pytesseract
            except ImportError:
                return None
            ocr_text = pytesseract.image_to_string(image, config="--psm 6")
        elif engine == "paddle":
            try:
                import numpy as np
            except ImportError:
                return None
            results = self.engine.ocr(np.array(image), cls=False)
            texts = []
            for line in results[0] if results else []:
                if len(line) >= 2:
                    texts.append(line[1][0])
            ocr_text = " ".join(texts)
        elif engine == "doctr":
            result = self.engine([image])
            texts = []
            for page_result in result.pages:
                for block in page_result.blocks:
                    for line in block.lines:
                        for word in line.words:
                            texts.append(word.value)
            ocr_text = " ".join(texts)
        else:
            return None

        ocr_tokens = set(tokenize_text(ocr_text))
        extracted_tokens = set(tokenize_text(" ".join(cell.text for cell in table.cells)))
        if not extracted_tokens:
            return OcrResult(overlap=0.0, token_count=0, engine=engine, device=self.device)
        overlap = len(extracted_tokens & ocr_tokens) / len(extracted_tokens)
        return OcrResult(overlap=overlap, token_count=len(ocr_tokens), engine=engine, device=self.device)

    def combine_confidence(self, base: float, ocr_overlap: float) -> float:
        return base * 0.7 + ocr_overlap * 0.3
