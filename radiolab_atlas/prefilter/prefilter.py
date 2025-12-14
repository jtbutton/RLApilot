from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from radiolab_atlas.models.ingestion import ResourceMetadata, TextChunk
from radiolab_atlas.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PrefilterDecision(str, Enum):
    SKIP = "SKIP"
    STUB_ONLY = "STUB_ONLY"
    LLM_CLASSIFY = "LLM_CLASSIFY"


@dataclass(frozen=True)
class PrefilterResult:
    chunk_id: str
    decision: PrefilterDecision
    reason_codes: List[str]
    score: Optional[float] = None
    features: Optional[Dict[str, float]] = None


@dataclass
class PrefilterConfig:
    min_chars: int = 80
    max_symbol_ratio: float = 0.25
    max_repetition_ratio: float = 0.35
    toc_heading_regex: str = r"(?i)^\s*(table of contents|contents)\s*$"
    references_heading_regex: str = r"(?i)^\s*(references|bibliography)\s*$"
    semantic_density_min: float = 0.18

    # If enabled later:
    enable_embedding_score: bool = False


class Prefilter:
    """
    Deterministic, explainable prefilter.
    Runs after chunking, before classification.

    Outputs per-chunk routing decision:
      - SKIP: drop from classification and embedding (still logged)
      - STUB_ONLY: bypass LLM; produce empty/stub classifications
      - LLM_CLASSIFY: eligible for batch LLM classification
    """

    def __init__(self, config: PrefilterConfig):
        self.config = config
        self._re_toc = re.compile(config.toc_heading_regex)
        self._re_refs = re.compile(config.references_heading_regex)

    def evaluate(self, chunks: List[TextChunk], resource_meta: ResourceMetadata) -> List[PrefilterResult]:
        results: List[PrefilterResult] = []

        for ch in chunks:
            decision, reasons, features, score = self._evaluate_one(ch)
            results.append(
                PrefilterResult(
                    chunk_id=ch.chunk_id,
                    decision=decision,
                    reason_codes=reasons,
                    features=features,
                    score=score,
                )
            )

        return results

    def _evaluate_one(self, chunk: TextChunk) -> tuple[PrefilterDecision, List[str], Dict[str, float], Optional[float]]:
        text = (chunk.text or "").strip()
        reasons: List[str] = []
        features: Dict[str, float] = {}

        # 1) too short
        n_chars = len(text)
        features["n_chars"] = float(n_chars)
        if n_chars < self.config.min_chars:
            reasons.append("TOO_SHORT")
            return PrefilterDecision.SKIP, reasons, features, None

        # 2) TOC / references headings (simple)
        first_line = text.splitlines()[0].strip() if text.splitlines() else text[:80].strip()
        if self._re_toc.match(first_line):
            reasons.append("TOC_HEADING")
            return PrefilterDecision.SKIP, reasons, features, None
        if self._re_refs.match(first_line):
            reasons.append("REFERENCES_HEADING")
            return PrefilterDecision.STUB_ONLY, reasons, features, None

        # 3) symbol ratio
        symbol_ratio = self._symbol_ratio(text)
        features["symbol_ratio"] = symbol_ratio
        if symbol_ratio > self.config.max_symbol_ratio:
            reasons.append("HIGH_SYMBOL_RATIO")
            return PrefilterDecision.STUB_ONLY, reasons, features, None

        # 4) repetition ratio (very rough)
        rep_ratio = self._repetition_ratio(text)
        features["repetition_ratio"] = rep_ratio
        if rep_ratio > self.config.max_repetition_ratio:
            reasons.append("HIGH_REPETITION_RATIO")
            return PrefilterDecision.STUB_ONLY, reasons, features, None

        # 5) semantic density (rough proxy)
        density = self._semantic_density(text)
        features["semantic_density"] = density
        if density < self.config.semantic_density_min:
            reasons.append("LOW_SEMANTIC_DENSITY")
            return PrefilterDecision.STUB_ONLY, reasons, features, None

        # Otherwise, keep for LLM
        reasons.append("PASS")
        return PrefilterDecision.LLM_CLASSIFY, reasons, features, density

    @staticmethod
    def _symbol_ratio(text: str) -> float:
        if not text:
            return 1.0
        symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
        return float(symbols) / float(max(len(text), 1))

    @staticmethod
    def _repetition_ratio(text: str) -> float:
        """
        Naive repetition: percent of tokens that are duplicates beyond first occurrence.
        """
        tokens = [t for t in re.split(r"\s+", text.lower()) if t]
        if not tokens:
            return 1.0
        seen = set()
        dup = 0
        for t in tokens:
            if t in seen:
                dup += 1
            else:
                seen.add(t)
        return float(dup) / float(len(tokens))

    @staticmethod
    def _semantic_density(text: str) -> float:
        """
        Proxy metric: ratio of alphabetic tokens to all tokens.
        """
        tokens = [t for t in re.split(r"\s+", text) if t]
        if not tokens:
            return 0.0
        alpha = sum(1 for t in tokens if re.search(r"[A-Za-z]", t))
        return float(alpha) / float(len(tokens))
