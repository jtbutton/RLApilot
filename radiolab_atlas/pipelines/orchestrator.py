from __future__ import annotations

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
    - load -> chunk -> classify -> validate -> write (neo4j/postgres/vector)
    - returns a structured IngestionRunReport
    """

    def __init__(self, settings: Settings, deps: OrchestratorDependencies) -> None:
        self.settings = settings
        self.deps = deps

    def ingest_documents(
        self,
        locations: Iterable[str],
        run_id: Optional[str] = None,
    ) -> IngestionRunReport:
        report = IngestionRunReport.start_new(run_id=run_id)

        for location in locations:
            try:
                logger.info("Starting ingestion for location=%s", location)
                resource_report = self._ingest_single(location)
                report.resource_reports.append(resource_report)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Fatal error ingesting %s", location)
                report.add_fatal_error(location=location, error=str(exc))

        report.finalize()
        return report

    def _ingest_single(self, location: str) -> ResourceIngestionReport:
        # 1. Load
        raw_doc, resource_meta = self._load_document(location)

        # 2. Chunk
        chunks = self._chunk_document(raw_doc, resource_meta)

        # 3. Classify
        classified_chunks = self._classify_chunks(chunks, resource_meta)

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

    def _classify_chunks(
        self,
        chunks: List[TextChunk],
        resource_meta: ResourceMetadata,
    ) -> List[ClassifiedChunk]:
        return self.deps.classifier.classify(chunks, resource_meta)

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
