from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Optional

from radiolab_atlas.config import Settings
from radiolab_atlas.models.ingestion import (
    RawDocument,
    ResourceMetadata,
    TextChunk,
    ClassifiedChunk,
)
from radiolab_atlas.models.graph import ValidatedGraphBatch
from radiolab_atlas.models.report import IngestionRunReport, ResourceIngestionReport
from radiolab_atlas.loaders.document_loader import DocumentLoader
from radiolab_atlas.chunkers.default_chunker import DefaultChunker
from radiolab_atlas.classifiers.ontology_classifier import OntologyClassifier
from radiolab_atlas.validators.ontology_validator import OntologyValidator, ValidationReport
from radiolab_atlas.writers.neo4j_writer import Neo4jWriter
from radiolab_atlas.writers.postgres_writer import PostgresWriter
from radiolab_atlas.writers.vector_writer import VectorWriter
from radiolab_atlas.utils.logging_utils import get_logger

# Phase 1B modules (new)
from radiolab_atlas.prefilter.prefilter import Prefilter, PrefilterConfig, PrefilterDecision
from radiolab_atlas.classifiers.batch_classifier import BatchClassifier, BatchConfig
from radiolab_atlas.budgets.budget_manager import BudgetManager, BudgetPolicy
from radiolab_atlas.telemetry.cost_logger import CostLogger

logger = get_logger(__name__)


@dataclass
class OrchestratorDependencies:
    """Container for dependency-injected subsystem instances."""

    document_loader: DocumentLoader
    chunker: DefaultChunker
    classifier: OntologyClassifier
    validator: OntologyValidator
    neo4j_writer: Neo4jWriter
    postgres_writer: PostgresWriter
    vector_writer: VectorWriter


class IngestionOrchestrator:
    """Coordinates the end-to-end ingestion pipeline for Radiolab Atlas.

    Pipeline:
    - load -> chunk -> prefilter -> (batch classify | stub) -> validate -> write
    - returns a structured IngestionRunReport
    """

    def __init__(self, settings: Settings, deps: OrchestratorDependencies) -> None:
        self.settings = settings
        self.deps = deps

        # Phase 1B: instantiate wrappers locally (do not replace stable modules)
        self.prefilter = Prefilter(
            PrefilterConfig(
                min_chars=settings.prefilter_min_chars,
            )
        )

        self.batch_classifier = BatchClassifier(
            base_classifier=deps.classifier,
            config=BatchConfig(
                batch_size=settings.batch_size,
                max_batch_chars=settings.batch_max_chars,
            ),
        )

        self.budget_manager = BudgetManager(
            policy=BudgetPolicy(
                max_cost_per_run_usd=settings.budget_max_cost_per_run_usd,
                max_cost_per_doc_usd=settings.budget_max_cost_per_doc_usd,
                max_cost_per_day_usd=settings.budget_max_cost_per_day_usd,
                max_calls_per_minute=settings.budget_max_calls_per_minute,
                throttle_seconds=settings.budget_throttle_seconds,
                downgrade_on_quota_error=True,
                dry_run=settings.budget_dry_run,
            )
        )

        self.cost_logger = CostLogger(output_path=settings.cost_log_path)

    def ingest_documents(
        self,
        locations: Iterable[str],
        run_id: Optional[str] = None,
    ) -> IngestionRunReport:
        report = IngestionRunReport.start_new(run_id=run_id)

        for location in locations:
            try:
                logger.info("Starting ingestion for location=%s", location)
                resource_report = self._ingest_single(location, run_id=report.run_id)
                report.resource_reports.append(resource_report)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Fatal error ingesting %s", location)
                report.add_fatal_error(location=location, error=str(exc))

        report.finalize()
        return report

    def _ingest_single(self, location: str, run_id: str) -> ResourceIngestionReport:
        # 1. Load
        raw_doc, resource_meta = self._load_document(location)

        # 2. Chunk
        chunks = self._chunk_document(raw_doc, resource_meta)

        # 3. Prefilter -> Classify (Phase 1B wiring)
        classified_chunks = self._prefilter_and_classify(chunks, resource_meta, run_id=run_id)

        # 4. Validate
        graph_batch, validation_report = self._validate_classification(
            classified_chunks, resource_meta
        )

        # 5. Write
        write_results = self._write_outputs(resource_meta, graph_batch, classified_chunks)

        # 6. Build resource report
        resource_report = self._build_resource_report(
            location=location,
            resource_meta=resource_meta,
            validation_report=validation_report,
            write_results=write_results,
        )

        logger.info("Completed ingestion for resource_id=%s", resource_meta.id)
        return resource_report

    def _load_document(self, location: str) -> tuple[RawDocument, ResourceMetadata]:
        return self.deps.document_loader.load(location)

    def _chunk_document(
        self,
        raw_doc: RawDocument,
        resource_meta: ResourceMetadata,
    ) -> List[TextChunk]:
        return self.deps.chunker.chunk(raw_doc, resource_meta)

    def _prefilter_and_classify(
        self,
        chunks: List[TextChunk],
        resource_meta: ResourceMetadata,
        run_id: str,
    ) -> List[ClassifiedChunk]:
        """
        Phase 1B wiring:
          Chunk -> Prefilter -> BatchClassifier -> BudgetManager -> CostLogger

        NOTE:
        - In stub mode, BatchClassifier delegates to existing classifier.
        - If classification_mode=llm but budget_dry_run=True, we will NOT call the LLM.
        """

        # If prefilter disabled, treat all as LLM_CLASSIFY for routing consistency.
        if getattr(self.settings, "prefilter_enabled", True):
            prefilter_results = self.prefilter.evaluate(chunks, resource_meta)
        else:
            prefilter_results = [
                # default route: eligible for LLM
                type("Tmp", (), {"chunk_id": c.chunk_id, "decision": PrefilterDecision.LLM_CLASSIFY})()
                for c in chunks
            ]

        decision_by_id = {r.chunk_id: PrefilterDecision(r.decision) for r in prefilter_results}
        counts = Counter([decision_by_id[c.chunk_id].value for c in chunks])

        # Emit prefilter rollup
        self.cost_logger.emit_prefilter_rollup(
            run_id=run_id,
            doc_id=resource_meta.id,
            counts_by_decision=dict(counts),
            metadata={"resource_title": resource_meta.title},
        )

        llm_chunks = [c for c in chunks if decision_by_id.get(c.chunk_id) == PrefilterDecision.LLM_CLASSIFY]
        stub_only_chunks = [c for c in chunks if decision_by_id.get(c.chunk_id) == PrefilterDecision.STUB_ONLY]
        skip_chunks = [c for c in chunks if decision_by_id.get(c.chunk_id) == PrefilterDecision.SKIP]

        classified: List[ClassifiedChunk] = []

        # --- IMPORTANT SAFETY GATE ---
        # If user sets classification_mode=llm but budget_dry_run=True, do not call LLM.
        classification_mode = str(getattr(self.settings, "classification_mode", "stub")).lower()

        classified: List[ClassifiedChunk] = []

        # --------------------------------------------
        # STUB MODE: classify ONCE for non-skipped
        # --------------------------------------------
        if classification_mode != "llm":
            to_classify = [c for c in chunks if decision_by_id.get(c.chunk_id) != PrefilterDecision.SKIP]
            if to_classify:
                classified.extend(self.deps.classifier.classify(to_classify, resource_meta))

            # Add empty for skipped so downstream stays consistent
            for ch in skip_chunks:
                classified.append(self._empty_classification(ch))

            # Telemetry
            self.cost_logger.emit_doc_rollup(
                run_id=run_id,
                doc_id=resource_meta.id,
                llm_calls=self.budget_manager.totals.llm_calls,
                tokens_in=self.budget_manager.totals.tokens_in,
                tokens_out=self.budget_manager.totals.tokens_out,
                cost_usd=self.budget_manager.totals.doc_cost_usd,
                metadata={
                    "chunks_total": len(chunks),
                    "counts_by_decision": dict(counts),
                    "mode": "stub_single_pass",
                },
            )
            return classified

        # --------------------------------------------
        # LLM MODE: keep Phase 1B routing
        # --------------------------------------------

        # Budget dry-run safety: do not call LLM
        if self.budget_manager.policy.dry_run:
            logger.info(
                "BudgetManager dry_run enabled: skipping LLM calls; producing empty classifications for LLM_CLASSIFY chunks."
            )
            for ch in llm_chunks:
                classified.append(self._empty_classification(ch))
        else:
            if llm_chunks:
                classified.extend(self.batch_classifier.classify_batches(llm_chunks, resource_meta))

        # Stub-only chunks
        if stub_only_chunks:
            classified.extend(self.deps.classifier.classify(stub_only_chunks, resource_meta))

        # Skipped chunks
        for ch in skip_chunks:
            classified.append(self._empty_classification(ch))

        # Telemetry
        self.cost_logger.emit_doc_rollup(
            run_id=run_id,
            doc_id=resource_meta.id,
            llm_calls=self.budget_manager.totals.llm_calls,
            tokens_in=self.budget_manager.totals.tokens_in,
            tokens_out=self.budget_manager.totals.tokens_out,
            cost_usd=self.budget_manager.totals.doc_cost_usd,
            metadata={
                "chunks_total": len(chunks),
                "counts_by_decision": dict(counts),
                "mode": "llm_routed",
            },
        )

        return classified


    def _validate_classification(
        self,
        classified_chunks: List[ClassifiedChunk],
        resource_meta: ResourceMetadata,
    ) -> tuple[ValidatedGraphBatch, ValidationReport]:
        return self.deps.validator.validate(classified_chunks, resource_meta)

    def _write_outputs(
        self,
        resource_meta: ResourceMetadata,
        graph_batch: ValidatedGraphBatch,
        classified_chunks: List[ClassifiedChunk],
    ):
        # NOTE: keep the current writer contract intact for now.
        neo4j_result = self.deps.neo4j_writer.write(graph_batch)
        postgres_result = self.deps.postgres_writer.write(resource_meta, classified_chunks)
        vector_result = self.deps.vector_writer.write(classified_chunks)

        return {
            "neo4j": neo4j_result,
            "postgres": postgres_result,
            "vector": vector_result,
        }

    def _build_resource_report(
        self,
        location: str,
        resource_meta: ResourceMetadata,
        validation_report: ValidationReport,
        write_results: dict,
    ) -> ResourceIngestionReport:
        return ResourceIngestionReport.from_components(
            location=location,
            resource_meta=resource_meta,
            validation_report=validation_report,
            write_results=write_results,
        )

    @staticmethod
    def _empty_classification(chunk: TextChunk) -> ClassifiedChunk:
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
