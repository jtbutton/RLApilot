from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple


FILES: Dict[str, str] = {
    "radiolab_atlas/prefilter/__init__.py": """\
from .prefilter import Prefilter, PrefilterConfig, PrefilterDecision, PrefilterResult
""",
    "radiolab_atlas/prefilter/prefilter.py": """\
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
    toc_heading_regex: str = r"(?i)^\\s*(table of contents|contents)\\s*$"
    references_heading_regex: str = r"(?i)^\\s*(references|bibliography)\\s*$"
    semantic_density_min: float = 0.18

    # If enabled later:
    enable_embedding_score: bool = False


class Prefilter:
    \"""
    Deterministic, explainable prefilter.
    Runs after chunking, before classification.

    Outputs per-chunk routing decision:
      - SKIP: drop from classification and embedding (still logged)
      - STUB_ONLY: bypass LLM; produce empty/stub classifications
      - LLM_CLASSIFY: eligible for batch LLM classification
    \"""

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
        \"""
        Naive repetition: percent of tokens that are duplicates beyond first occurrence.
        \"""
        tokens = [t for t in re.split(r"\\s+", text.lower()) if t]
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
        \"""
        Proxy metric: ratio of alphabetic tokens to all tokens.
        \"""
        tokens = [t for t in re.split(r"\\s+", text) if t]
        if not tokens:
            return 0.0
        alpha = sum(1 for t in tokens if re.search(r"[A-Za-z]", t))
        return float(alpha) / float(len(tokens))
""",
    "radiolab_atlas/classifiers/batch_classifier.py": """\
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from radiolab_atlas.classifiers.ontology_classifier import OntologyClassifier
from radiolab_atlas.models.ingestion import ClassifiedChunk, ResourceMetadata, TextChunk
from radiolab_atlas.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class BatchConfig:
    batch_size: int = 6
    max_batch_chars: int = 30000  # safety cap to avoid huge prompts
    require_chunk_id_keyed_output: bool = True


class BatchClassifier:
    \"""
    Wraps existing OntologyClassifier without changing it.
    Batches multiple chunks into one LLM call when classifier is in LLM mode.

    In stub mode, delegates to base_classifier.classify().
    \"""

    def __init__(self, base_classifier: OntologyClassifier, config: BatchConfig):
        self.base = base_classifier
        self.config = config

    def make_batches(self, chunks: List[TextChunk]) -> List[List[TextChunk]]:
        batches: List[List[TextChunk]] = []
        cur: List[TextChunk] = []
        cur_chars = 0

        for ch in chunks:
            ch_len = len(ch.text or "")
            if cur and (len(cur) >= self.config.batch_size or cur_chars + ch_len > self.config.max_batch_chars):
                batches.append(cur)
                cur = []
                cur_chars = 0

            cur.append(ch)
            cur_chars += ch_len

        if cur:
            batches.append(cur)

        return batches

    def classify_batches(self, chunks: List[TextChunk], resource_meta: ResourceMetadata) -> List[ClassifiedChunk]:
        mode = getattr(self.base, "mode", "stub")
        if str(mode).lower() == "stub":
            return self.base.classify(chunks, resource_meta)

        output_by_id: Dict[str, ClassifiedChunk] = {}

        for batch_i, batch in enumerate(self.make_batches(chunks)):
            try:
                raw = self._classify_batch_llm(batch, resource_meta, batch_i=batch_i)
                mapped = self._map_batch_json_to_results(batch, raw)
                for cc in mapped:
                    output_by_id[cc.chunk.chunk_id] = cc
            except Exception as exc:  # noqa: BLE001
                logger.warning("Batch LLM classify failed (batch=%s). Falling back to empty classifications. Error=%s", batch_i, exc)
                for ch in batch:
                    output_by_id[ch.chunk_id] = self._empty_classification(ch)

        return [output_by_id.get(ch.chunk_id) or self._empty_classification(ch) for ch in chunks]

    def _classify_batch_llm(self, batch: List[TextChunk], resource_meta: ResourceMetadata, batch_i: int) -> Dict[str, Any]:
        llm_client = getattr(self.base, "llm_client", None)
        if llm_client is None:
            raise RuntimeError("Base classifier has no llm_client; cannot batch classify in LLM mode.")

        system_prompt = self._build_system_prompt_for_batch()
        user_prompt = self._build_user_prompt_for_batch(batch, resource_meta)

        raw_text = llm_client.classify_chunk(system_prompt=system_prompt, user_prompt=user_prompt)
        data = llm_client.parse_json_response(raw_text)
        return data

    def _build_system_prompt_for_batch(self) -> str:
        rubric_text = getattr(self.base, "rubric_text", "")
        return (
            "You classify multiple text chunks into an existing ontology. "
            "Return JSON only. No prose.\\n\\n"
            "Rubric:\\n"
            f"{rubric_text}\\n\\n"
            "Return JSON with this schema:\\n"
            "{\\n"
            '  "results": {\\n'
            '    "<chunk_id>": {\\n'
            '      "concept_ids": [string],\\n'
            '      "competency_ids": [string],\\n'
            '      "scenario_ids": [string],\\n'
            '      "role_ids": [string],\\n'
            '      "instrument_ids": [string],\\n'
            '      "network_program_ids": [string],\\n'
            '      "candidate_concept_labels": [string]\\n'
            "    }\\n"
            "  }\\n"
            "}\\n"
        )

    def _build_user_prompt_for_batch(self, batch: List[TextChunk], resource_meta: ResourceMetadata) -> str:
        vocab_snapshot = getattr(self.base, "vocab_snapshot", {})
        vocab_json = json.dumps(vocab_snapshot, indent=2)

        payload = {
            "resource_id": resource_meta.id,
            "title": resource_meta.title,
            "ontology_vocabulary_snapshot": vocab_snapshot,
            "chunks": [
                {"chunk_id": ch.chunk_id, "span": ch.span, "text": ch.text}
                for ch in batch
            ],
        }

        return (
            "Use the following ontology vocabulary snapshot (valid IDs):\\n"
            f"{vocab_json}\\n\\n"
            "Classify each chunk. Output must be keyed by chunk_id.\\n\\n"
            f"INPUT:\\n{json.dumps(payload, indent=2)}"
        )

    def _map_batch_json_to_results(self, batch: List[TextChunk], data: Dict[str, Any]) -> List[ClassifiedChunk]:
        results = data.get("results")
        if not isinstance(results, dict):
            raise ValueError("Batch response missing 'results' dict.")

        out: List[ClassifiedChunk] = []
        by_id = {c.chunk_id: c for c in batch}

        for chunk_id, payload in results.items():
            if chunk_id not in by_id:
                continue
            if not isinstance(payload, dict):
                out.append(self._empty_classification(by_id[chunk_id]))
                continue
            out.append(self._classified_from_payload(by_id[chunk_id], payload))

        for ch in batch:
            if not any(cc.chunk.chunk_id == ch.chunk_id for cc in out):
                out.append(self._empty_classification(ch))

        return out

    @staticmethod
    def _list_of_str(payload: Dict[str, Any], key: str) -> List[str]:
        v = payload.get(key, [])
        if not isinstance(v, list):
            return []
        return [str(x) for x in v]

    def _classified_from_payload(self, chunk: TextChunk, payload: Dict[str, Any]) -> ClassifiedChunk:
        return ClassifiedChunk(
            chunk=chunk,
            concept_ids=self._list_of_str(payload, "concept_ids"),
            competency_ids=self._list_of_str(payload, "competency_ids"),
            scenario_ids=self._list_of_str(payload, "scenario_ids"),
            role_ids=self._list_of_str(payload, "role_ids"),
            instrument_ids=self._list_of_str(payload, "instrument_ids"),
            network_program_ids=self._list_of_str(payload, "network_program_ids"),
            candidate_concept_labels=self._list_of_str(payload, "candidate_concept_labels"),
        )

    def _empty_classification(self, chunk: TextChunk) -> ClassifiedChunk:
        return ClassifiedChunk(
            chunk=chunk,
            concept_ids=[],
            competency_ids=[],
            scenario_ids=[],
            role_ids=[],
            instrument_ids=[],
            network_program_ids=[],
            candidate_concept_labels=[],
        )
""",
    "radiolab_atlas/budgets/__init__.py": """\
from .budget_manager import (
    BudgetManager,
    BudgetPolicy,
    BudgetDecision,
    UsageTotals,
    TokenEstimator,
)
""",
    "radiolab_atlas/budgets/budget_manager.py": """\
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from radiolab_atlas.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BudgetDecision(str, Enum):
    ALLOW = "ALLOW"
    THROTTLE = "THROTTLE"
    DOWNGRADE = "DOWNGRADE"


@dataclass
class BudgetPolicy:
    max_cost_per_run_usd: float = 2.00
    max_cost_per_doc_usd: float = 0.50
    max_cost_per_day_usd: float = 5.00

    max_calls_per_minute: int = 30
    throttle_seconds: float = 2.0

    downgrade_on_quota_error: bool = True
    dry_run: bool = False


@dataclass
class UsageTotals:
    run_cost_usd: float = 0.0
    doc_cost_usd: float = 0.0
    day_cost_usd: float = 0.0
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0


class TokenEstimator:
    def estimate_tokens(self, text: str, model: str) -> int:
        return max(1, int(len(text) / 4))


class BudgetManager:
    def __init__(self, policy: BudgetPolicy, estimator: Optional[TokenEstimator] = None):
        self.policy = policy
        self.estimator = estimator or TokenEstimator()
        self.totals = UsageTotals()
        self._call_timestamps: List[float] = []
        self._llm_disabled: bool = False

    def llm_disabled(self) -> bool:
        return self._llm_disabled

    def disable_llm(self) -> None:
        self._llm_disabled = True

    def preflight_batch(
        self,
        model: str,
        prompt_text: str,
        expected_output_tokens: int,
        doc_id: str,
        run_id: str,
        price_table: Optional[Dict[str, Any]] = None,
    ) -> Tuple[BudgetDecision, Dict[str, Any]]:
        if self._llm_disabled:
            return BudgetDecision.DOWNGRADE, {"reason": "LLM_DISABLED"}

        decision = self._rate_limit_decision()
        if decision != BudgetDecision.ALLOW:
            return decision, {"reason": "RATE_LIMIT"}

        est_in = self.estimator.estimate_tokens(prompt_text, model=model)
        est_out = int(expected_output_tokens)
        est_cost = self._estimate_cost_usd(model, est_in, est_out, price_table=price_table)

        telemetry = {
            "estimated_tokens_in": est_in,
            "estimated_tokens_out": est_out,
            "estimated_cost_usd": est_cost,
            "doc_id": doc_id,
            "run_id": run_id,
        }

        if self.totals.run_cost_usd + est_cost > self.policy.max_cost_per_run_usd:
            return BudgetDecision.DOWNGRADE, {**telemetry, "reason": "RUN_BUDGET_EXCEEDED"}
        if self.totals.doc_cost_usd + est_cost > self.policy.max_cost_per_doc_usd:
            return BudgetDecision.DOWNGRADE, {**telemetry, "reason": "DOC_BUDGET_EXCEEDED"}
        if self.totals.day_cost_usd + est_cost > self.policy.max_cost_per_day_usd:
            return BudgetDecision.DOWNGRADE, {**telemetry, "reason": "DAY_BUDGET_EXCEEDED"}

        if self.policy.dry_run:
            return BudgetDecision.DOWNGRADE, {**telemetry, "reason": "DRY_RUN_ESTIMATE_ONLY"}

        return BudgetDecision.ALLOW, telemetry

    def record_actual(
        self,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        doc_id: str,
        run_id: str,
    ) -> None:
        self.totals.llm_calls += 1
        self.totals.tokens_in += tokens_in
        self.totals.tokens_out += tokens_out
        self.totals.run_cost_usd += cost_usd
        self.totals.doc_cost_usd += cost_usd
        self.totals.day_cost_usd += cost_usd
        self._call_timestamps.append(time.time())

    def handle_quota_or_rate_error(self, exc: Exception) -> BudgetDecision:
        msg = str(exc).lower()
        if self.policy.downgrade_on_quota_error and ("insufficient_quota" in msg or "exceeded your current quota" in msg):
            self.disable_llm()
            return BudgetDecision.DOWNGRADE
        return BudgetDecision.THROTTLE

    def _rate_limit_decision(self) -> BudgetDecision:
        now = time.time()
        window = [t for t in self._call_timestamps if now - t < 60.0]
        self._call_timestamps = window

        if len(window) >= self.policy.max_calls_per_minute:
            time.sleep(self.policy.throttle_seconds)
            return BudgetDecision.THROTTLE
        return BudgetDecision.ALLOW

    @staticmethod
    def _estimate_cost_usd(
        model: str,
        tokens_in: int,
        tokens_out: int,
        price_table: Optional[Dict[str, Any]] = None,
    ) -> float:
        if not price_table or model not in price_table:
            return 0.0

        p = price_table[model]
        in_cost = (tokens_in / 1000.0) * float(p["input_per_1k"])
        out_cost = (tokens_out / 1000.0) * float(p["output_per_1k"])
        return in_cost + out_cost
""",
    "radiolab_atlas/telemetry/__init__.py": """\
from .cost_logger import CostLogger, CostEvent
""",
    "radiolab_atlas/telemetry/cost_logger.py": """\
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from radiolab_atlas.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class CostEvent:
    timestamp: float
    run_id: str
    doc_id: str
    event_type: str
    batch_id: Optional[str] = None
    chunk_ids: Optional[List[str]] = None

    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    cost_usd: Optional[float] = None

    estimated_tokens_in: Optional[int] = None
    estimated_tokens_out: Optional[int] = None
    estimated_cost_usd: Optional[float] = None

    metadata: Optional[Dict[str, Any]] = None


class CostLogger:
    def __init__(self, output_path: str = "data/logs/cost_events.jsonl"):
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: CostEvent) -> None:
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(event)) + "\\n")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to write cost event. Error=%s", exc)

    def emit_prefilter_rollup(
        self,
        run_id: str,
        doc_id: str,
        counts_by_decision: Dict[str, int],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.emit(
            CostEvent(
                timestamp=time.time(),
                run_id=run_id,
                doc_id=doc_id,
                event_type="prefilter",
                metadata={"counts_by_decision": counts_by_decision, **(metadata or {})},
            )
        )

    def emit_doc_rollup(
        self,
        run_id: str,
        doc_id: str,
        llm_calls: int,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.emit(
            CostEvent(
                timestamp=time.time(),
                run_id=run_id,
                doc_id=doc_id,
                event_type="doc_rollup",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd,
                metadata=metadata,
            )
        )
""",
    "radiolab_atlas/writers/neo4j_writer.py": """\
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from neo4j import GraphDatabase

from radiolab_atlas.models.graph import GraphNode, GraphRelationship, ValidatedGraphBatch
from radiolab_atlas.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Neo4jWriteSummary:
    nodes_merged: int
    relationships_merged: int
    constraints_created: int
    status: str


class Neo4jWriter:
    \"""
    Production MERGE-based Neo4j writer (skeleton).
    \"""

    def __init__(self, uri: str, user: str, password: str, database: Optional[str] = None):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self) -> None:
        self.driver.close()

    def ensure_constraints(self) -> int:
        constraints_created = 0
        statements = [
            "CREATE CONSTRAINT resource_id IF NOT EXISTS FOR (n:Resource) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (n:Concept) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT scenario_id IF NOT EXISTS FOR (n:Scenario) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT competency_id IF NOT EXISTS FOR (n:Competency) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT role_id IF NOT EXISTS FOR (n:Role) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT instrument_id IF NOT EXISTS FOR (n:Instrument) REQUIRE n.id IS UNIQUE",
        ]
        try:
            with self.driver.session(database=self.database) as session:
                for stmt in statements:
                    session.run(stmt)
                    constraints_created += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Neo4j constraint creation failed or not permitted. Error=%s", exc)
        return constraints_created

    def write(self, batch: ValidatedGraphBatch, provenance: Dict[str, Any]) -> Neo4jWriteSummary:
        constraints_created = self.ensure_constraints()
        nodes_merged = 0
        rels_merged = 0
        ingested_at = provenance.get("ingested_at") or time.time()

        try:
            with self.driver.session(database=self.database) as session:
                for node in batch.nodes:
                    self._merge_node(session, node, provenance, ingested_at)
                    nodes_merged += 1

                for rel in batch.relationships:
                    self._merge_relationship(session, rel, provenance, ingested_at)
                    rels_merged += 1

            return Neo4jWriteSummary(
                nodes_merged=nodes_merged,
                relationships_merged=rels_merged,
                constraints_created=constraints_created,
                status="ok",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Neo4j write failed")
            return Neo4jWriteSummary(
                nodes_merged=nodes_merged,
                relationships_merged=rels_merged,
                constraints_created=constraints_created,
                status=f"error: {exc}",
            )

    @staticmethod
    def _merge_node(session, node: GraphNode, provenance: Dict[str, Any], ingested_at: float) -> None:
        label = node.label
        props = dict(node.properties or {})
        props.setdefault("id", node.id)

        cypher = f\"\"\"
        MERGE (n:`{label}` {{ id: $id }})
        SET n += $props
        SET n.source_doc = $source_doc,
            n.run_id = $run_id,
            n.ingested_at = $ingested_at
        \"\"\"
        session.run(
            cypher,
            id=node.id,
            props=props,
            source_doc=provenance.get("source_doc"),
            run_id=provenance.get("run_id"),
            ingested_at=ingested_at,
        )

    @staticmethod
    def _merge_relationship(session, rel: GraphRelationship, provenance: Dict[str, Any], ingested_at: float) -> None:
        rel_type = rel.type
        props = dict(rel.properties or {})

        cypher = f\"\"\"
        MATCH (a {{ id: $start_id }})
        MATCH (b {{ id: $end_id }})
        MERGE (a)-[r:`{rel_type}`]->(b)
        SET r += $props
        SET r.source_doc = $source_doc,
            r.run_id = $run_id,
            r.ingested_at = $ingested_at
        \"\"\"
        session.run(
            cypher,
            start_id=rel.start_id,
            end_id=rel.end_id,
            props=props,
            source_doc=provenance.get("source_doc"),
            run_id=provenance.get("run_id"),
            ingested_at=ingested_at,
        )
""",
}


def write_file(path: Path, content: str, overwrite: bool) -> Tuple[bool, str]:
    if path.exists() and not overwrite:
        return False, "exists"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return True, "written"


def main() -> None:
    parser = argparse.ArgumentParser(description="Add Phase 1B skeleton modules to Radiolab Atlas.")
    parser.add_argument("--root", default=".", help="Repo root (default: current directory)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be written without writing")
    args = parser.parse_args()

    root = Path(args.root).resolve()

    planned = []
    for rel_path, content in FILES.items():
        planned.append((root / rel_path, content))

    if args.dry_run:
        print("DRY RUN: planned writes:")
        for p, _ in planned:
            status = "would write" if (not p.exists() or args.overwrite) else "skip (exists)"
            print(f" - {status}: {p}")
        return

    wrote = 0
    skipped = 0
    for p, content in planned:
        ok, why = write_file(p, content, overwrite=args.overwrite)
        if ok:
            wrote += 1
            print(f"WROTE: {p}")
        else:
            skipped += 1
            print(f"SKIP:  {p} ({why})")

    print(f"\nDone. wrote={wrote} skipped={skipped}")
    if skipped:
        print("Tip: re-run with --overwrite if you want to replace existing files.")


if __name__ == "__main__":
    main()
