from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

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
    # --- basic thresholds ---
    min_chars: int = 80
    min_lines: int = 2

    # --- density thresholds ---
    min_alpha_ratio: float = 0.55          # alphabetic chars / all non-space chars
    min_word_ratio: float = 0.35           # "word-like tokens" / tokens
    max_symbol_ratio: float = 0.25
    max_digit_ratio: float = 0.35

    # --- TOC detection ---
    toc_dot_leader_min_lines: int = 3      # lines with ".... 12"
    toc_page_num_line_ratio: float = 0.35  # proportion of lines ending in page-like numbers

    # --- table-like detection ---
    table_like_min_pipes: int = 2          # markdown-ish tables
    table_like_min_runs_of_spaces: int = 3 # multi-column alignment
    table_like_min_numeric_tokens: int = 6

    # --- header/footer repetition ---
    header_footer_window_lines: int = 2    # evaluate first/last N lines of chunk
    repeat_min_count: int = 5              # seen >= N times across doc => boilerplate
    repeat_min_len: int = 12               # avoid tiny repeats like "Page 1"

    # --- reference section heuristics ---
    references_heading_regex: str = r"(?i)^\s*(references|bibliography|literature cited)\s*$"
    # patterns common to citations
    citation_line_regex: str = r"(?i)\b(doi:|et al\.|vol\.|pp\.|isbn|issn)\b|^\s*\[\d+\]\s+"

    # When a "references section" is detected, mark subsequent chunks as STUB_ONLY.
    enable_section_state: bool = True


class Prefilter:
    """
    Prefilter v1 (deterministic, explainable).
    - Uses doc-level analysis to detect repeated headers/footers.
    - Detects TOC-like and table-like chunks.
    - Detects references section and downgrades.
    """

    def __init__(self, config: PrefilterConfig):
        self.config = config
        self._re_refs = re.compile(config.references_heading_regex)
        self._re_cite = re.compile(config.citation_line_regex)

    def evaluate(self, chunks: List[TextChunk], resource_meta: ResourceMetadata) -> List[PrefilterResult]:
        # Build repetition map for header/footer strings across all chunks
        repeats = self._build_repetition_index(chunks)

        in_references_section = False
        results: List[PrefilterResult] = []

        for ch in chunks:
            text = (ch.text or "").strip()

            # section-state tracking (optional)
            if self.config.enable_section_state:
                if self._looks_like_references_heading(text):
                    in_references_section = True

            decision, reasons, features, score = self._evaluate_one(
                chunk=ch,
                repeats=repeats,
                in_references_section=in_references_section,
            )

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

    # -----------------------
    # Doc-level repetition index
    # -----------------------

    def _build_repetition_index(self, chunks: List[TextChunk]) -> Dict[str, int]:
        """
        Returns count map for normalized candidate header/footer strings.
        We index first/last N lines of each chunk.
        """
        counts: Dict[str, int] = {}
        N = self.config.header_footer_window_lines

        for ch in chunks:
            lines = self._nonempty_lines(ch.text or "")
            if not lines:
                continue

            head = " ".join(lines[:N]).strip()
            tail = " ".join(lines[-N:]).strip()

            for s in (head, tail):
                key = self._normalize_repeat_string(s)
                if len(key) < self.config.repeat_min_len:
                    continue
                counts[key] = counts.get(key, 0) + 1

        return counts

    @staticmethod
    def _normalize_repeat_string(s: str) -> str:
        # Normalize digits (page numbers), whitespace, case
        s = s.strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"\d+", "<n>", s)
        return s

    # -----------------------
    # Chunk evaluation
    # -----------------------

    def _evaluate_one(
        self,
        chunk: TextChunk,
        repeats: Dict[str, int],
        in_references_section: bool,
    ) -> Tuple[PrefilterDecision, List[str], Dict[str, float], Optional[float]]:
        text = (chunk.text or "").strip()
        lines = self._nonempty_lines(text)

        reasons: List[str] = []
        features: Dict[str, float] = {}

        # Basic length checks
        features["n_chars"] = float(len(text))
        features["n_lines"] = float(len(lines))

        if len(text) < self.config.min_chars or len(lines) < self.config.min_lines:
            reasons.append("TOO_SHORT")
            return PrefilterDecision.SKIP, reasons, features, None

        # Repeated header/footer boilerplate
        if self._is_repeated_header_footer(lines, repeats):
            reasons.append("HEADER_FOOTER_REPEAT")
            return PrefilterDecision.SKIP, reasons, features, None

        # If we're in a references section, default to STUB_ONLY unless it clearly contains procedural content
        if in_references_section:
            reasons.append("REFERENCES_SECTION")
            # Still allow LLM for unusually content-rich chunks inside refs section (rare)
            if self._semantic_density_score(text) > 0.75 and not self._is_citation_heavy(lines):
                reasons.append("OVERRIDE_CONTENT_RICH")
                return PrefilterDecision.LLM_CLASSIFY, reasons, features, 0.75
            return PrefilterDecision.STUB_ONLY, reasons, features, 0.25

        # TOC detection
        toc_score, toc_reasons = self._toc_signals(lines)
        features["toc_score"] = toc_score
        if toc_score >= 1.0:
            reasons.extend(toc_reasons)
            return PrefilterDecision.SKIP, reasons, features, None

        # Table-like detection
        table_score, table_reasons = self._table_signals(text, lines)
        features["table_score"] = table_score
        if table_score >= 1.0:
            reasons.extend(table_reasons)
            return PrefilterDecision.STUB_ONLY, reasons, features, 0.2

        # Density / symbol / digit checks
        alpha_ratio = self._alpha_ratio(text)
        digit_ratio = self._digit_ratio(text)
        symbol_ratio = self._symbol_ratio(text)
        word_ratio = self._word_ratio(text)

        features.update(
            {
                "alpha_ratio": alpha_ratio,
                "digit_ratio": digit_ratio,
                "symbol_ratio": symbol_ratio,
                "word_ratio": word_ratio,
            }
        )

        if symbol_ratio > self.config.max_symbol_ratio:
            reasons.append("HIGH_SYMBOL_RATIO")
            return PrefilterDecision.STUB_ONLY, reasons, features, 0.2

        if digit_ratio > self.config.max_digit_ratio and alpha_ratio < 0.45:
            reasons.append("HIGH_DIGIT_LOW_ALPHA")
            return PrefilterDecision.STUB_ONLY, reasons, features, 0.2

        if alpha_ratio < self.config.min_alpha_ratio and word_ratio < self.config.min_word_ratio:
            reasons.append("LOW_ALPHA_LOW_WORD")
            return PrefilterDecision.STUB_ONLY, reasons, features, 0.2

        # Citation-heavy chunk (even if not explicitly in references section)
        if self._is_citation_heavy(lines):
            reasons.append("CITATION_HEAVY")
            return PrefilterDecision.STUB_ONLY, reasons, features, 0.25

        # Otherwise: LLM-worthy
        density = self._semantic_density_score(text)
        features["semantic_density"] = density
        reasons.append("PASS")
        return PrefilterDecision.LLM_CLASSIFY, reasons, features, density

    # -----------------------
    # Signals
    # -----------------------

    def _is_repeated_header_footer(self, lines: List[str], repeats: Dict[str, int]) -> bool:
        N = self.config.header_footer_window_lines
        head = self._normalize_repeat_string(" ".join(lines[:N]))
        tail = self._normalize_repeat_string(" ".join(lines[-N:]))

        head_ct = repeats.get(head, 0)
        tail_ct = repeats.get(tail, 0)

        return head_ct >= self.config.repeat_min_count or tail_ct >= self.config.repeat_min_count

    def _looks_like_references_heading(self, text: str) -> bool:
        lines = self._nonempty_lines(text)
        if not lines:
            return False
        # heading on its own line
        return bool(self._re_refs.match(lines[0])) or bool(self._re_refs.match(text.strip()))

    def _is_citation_heavy(self, lines: List[str]) -> bool:
        if not lines:
            return False
        cite_like = 0
        for ln in lines[: min(len(lines), 12)]:  # check first few lines
            if self._re_cite.search(ln):
                cite_like += 1
            # years pattern
            if re.search(r"\b(19|20)\d{2}\b", ln):
                cite_like += 1
        return cite_like >= 4

    def _toc_signals(self, lines: List[str]) -> Tuple[float, List[str]]:
        """
        Score >= 1.0 => TOC-like.
        """
        reasons: List[str] = []
        dot_leader_lines = 0
        page_num_end_lines = 0

        for ln in lines:
            s = ln.strip()
            if not s:
                continue

            # dot leaders like "Section ....... 12"
            if re.search(r"\.{3,}\s*\d+\s*$", s):
                dot_leader_lines += 1

            # line ends with page number or roman numeral
            if re.search(r"\s(\d{1,4}|[ivxlcdm]{1,8})\s*$", s.lower()):
                page_num_end_lines += 1

        score = 0.0
        if dot_leader_lines >= self.config.toc_dot_leader_min_lines:
            score += 1.0
            reasons.append("TOC_DOT_LEADER")
        if (page_num_end_lines / max(len(lines), 1)) >= self.config.toc_page_num_line_ratio:
            score += 0.75
            reasons.append("TOC_PAGE_NUM_DENSITY")

        # additional TOC keyword boost
        if lines and re.match(r"(?i)^\s*(contents|table of contents)\s*$", lines[0].strip()):
            score += 1.0
            reasons.append("TOC_HEADING")

        return score, reasons

    def _table_signals(self, text: str, lines: List[str]) -> Tuple[float, List[str]]:
        """
        Score >= 1.0 => table-like.
        """
        reasons: List[str] = []
        score = 0.0

        # Many pipe characters suggests markdown table or extracted table text
        pipe_count = text.count("|")
        if pipe_count >= self.config.table_like_min_pipes:
            score += 0.6
            reasons.append("TABLE_PIPES")

        # Many runs of multiple spaces suggests multi-column alignment
        space_runs = len(re.findall(r"\s{3,}", text))
        if space_runs >= self.config.table_like_min_runs_of_spaces:
            score += 0.6
            reasons.append("TABLE_SPACING")

        # Many numeric tokens in short span suggests tabular
        nums = re.findall(r"\b\d+(\.\d+)?\b", text)
        if len(nums) >= self.config.table_like_min_numeric_tokens:
            score += 0.6
            reasons.append("TABLE_NUMERIC_DENSITY")

        return score, reasons

    # -----------------------
    # Feature computations
    # -----------------------

    @staticmethod
    def _nonempty_lines(text: str) -> List[str]:
        return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]

    @staticmethod
    def _alpha_ratio(text: str) -> float:
        s = re.sub(r"\s+", "", text or "")
        if not s:
            return 0.0
        alpha = sum(1 for c in s if c.isalpha())
        return float(alpha) / float(len(s))

    @staticmethod
    def _digit_ratio(text: str) -> float:
        s = re.sub(r"\s+", "", text or "")
        if not s:
            return 0.0
        digits = sum(1 for c in s if c.isdigit())
        return float(digits) / float(len(s))

    @staticmethod
    def _symbol_ratio(text: str) -> float:
        s = re.sub(r"\s+", "", text or "")
        if not s:
            return 0.0
        sym = sum(1 for c in s if not c.isalnum())
        return float(sym) / float(len(s))

    @staticmethod
    def _word_ratio(text: str) -> float:
        tokens = [t for t in re.split(r"\s+", text or "") if t]
        if not tokens:
            return 0.0
        word_like = sum(1 for t in tokens if re.search(r"[A-Za-z]{2,}", t))
        return float(word_like) / float(len(tokens))

    def _semantic_density_score(self, text: str) -> float:
        """
        Deterministic proxy score (0..1):
        higher when alpha_ratio + word_ratio are high and symbol/digit ratios are low.
        """
        a = self._alpha_ratio(text)
        w = self._word_ratio(text)
        d = self._digit_ratio(text)
        s = self._symbol_ratio(text)

        # bounded combination
        score = (0.55 * a) + (0.55 * w) - (0.30 * d) - (0.25 * s)
        return max(0.0, min(1.0, score))
