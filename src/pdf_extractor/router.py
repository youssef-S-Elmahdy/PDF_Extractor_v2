"""Document-class routing based on content heuristics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class RouteResult:
    class_name: str
    confidence: float


class DocumentRouter:
    def __init__(self) -> None:
        self.classes = {
            "cash_earnings_ledger": [
                "ledger",
                "cash earnings",
                "cash ledger",
                "transaction",
                "balance",
            ],
            "income_statement": [
                "income statement",
                "statement of income",
                "profit and loss",
                "net income",
            ],
            "balance_sheet": [
                "balance sheet",
                "assets",
                "liabilities",
                "equity",
            ],
            "cash_flow": [
                "cash flow",
                "operating activities",
                "investing activities",
                "financing activities",
            ],
        }

    def classify(self, page_texts: List[str]) -> RouteResult:
        text = " ".join(page_texts).lower()
        scores = {"generic": 0}
        for class_name, keywords in self.classes.items():
            hits = sum(1 for keyword in keywords if keyword in text)
            scores[class_name] = hits

        best_class = max(scores, key=scores.get)
        max_hits = scores[best_class]
        if best_class == "generic" or max_hits == 0:
            return RouteResult("generic", 0.0)
        confidence = min(1.0, max_hits / 4.0)
        return RouteResult(best_class, confidence)
