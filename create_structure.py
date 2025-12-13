from __future__ import annotations

from pathlib import Path

# Map of relative file paths -> file contents
FILES = {
    # Package root
    "radiolab_atlas/__init__.py": """"""Radiolab Atlas ingestion package."""\n""",
    "radiolab_atlas/config.py": r"""from pydantic import BaseSettings


class Settings(BaseSettings):
    environment: str = "dev"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # Postgres
    postgres_dsn: str = "postgresql+psycopg2://user:password@localhost/radiolab"

    # Chroma
    chroma_path: str = "./chroma_db"

    # LLM
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    embedding_model_name: str = "text-embedding-3-large"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 200

    class Config:
        env_prefix = "RADIOLAB_ATLAS_"
        env_file = ".env"
""",

    # utils
    "radiolab_atlas/utils/__init__.py": "",
    "radiolab_atlas/utils/logging_utils.py": r"""import logging
import sys


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
""",

    # models
    "radiolab_atlas/models/__init__.py": "",
    "radiolab_atlas/models/ingestion.py": r"""from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel


class ResourceMetadata(BaseModel):
    """Minimal metadata for a Resource node, aligned with ontology fields."""

    id: str
    title: str
    resource_type: str
    summary: Optional[str] = None
    source_program: Optional[str] = None
    version: Optional[str] = None
    date_published: Optional[str] = None  # ISO string
    file_type: Optional[str] = None
    authors: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    ontology_version: Optional[str] = None


class RawDocument(BaseModel):
    resource_id: str
    location: str
    text: str
    structure_hints: Optional[dict] = None  # pages/slides/headings/etc.


class TextChunk(BaseModel):
    chunk_id: str
    resource_id: str
    text: str
    order_index: int
    span: Optional[dict] = None  # e.g. {\"pages\": [1, 2]} or {\"slide\": 3}


class ClassifiedChunk(BaseModel):
    chunk: TextChunk
    concept_ids: List[str] = []
    competency_ids: List[str] = []
    scenario_ids: List[str] = []
    role_ids: List[str] = []
    instrument_ids: List[str] = []
    network_program_ids: List[str] = []
    candidate_concept_labels: List[str] = []
""",
    "radiolab_atlas/models/graph.py": r"""from typing import Any, Dict, List
from pydantic import BaseModel


class GraphNode(BaseModel):
    id: str
    label: str
    properties: Dict[str, Any]


class GraphRelationship(BaseModel):
    id: str
    type: str
    start_id: str
    end_id: str
    properties: Dict[str, Any] = {}


class ValidatedGraphBatch(BaseModel):
    nodes: List[GraphNode]
    relationships: List[GraphRelationship]
""",
    "radiolab_atlas/models/report.py": r"""from typing import List, Optional
from pydantic import BaseModel


class ResourceIngestionReport(BaseModel):
    location: str
    resource_id: Optional[str]
    status: str
    validation_warnings: List[str] = []
    validation_errors: List[str] = []
    neo4j_summary: Optional[dict] = None
    postgres_summary: Optional[dict] = None
    vector_summary: Optional[dict] = None

    @classmethod
    def from_components(cls, location, resource_meta, validation_report, write_results):
        status = "success"
        if validation_report.errors:
            status = "partial"

        return cls(
            location=location,
            resource_id=resource_meta.id,
            status=status,
            validation_warnings=validation_report.warnings,
            validation_errors=validation_report.errors,
            neo4j_summary=write_results.get("neo4j"),
            postgres_summary=write_results.get("postgres"),
            vector_summary=write_results.get("vector"),
        )


class IngestionRunReport(BaseModel):
    run_id: Optional[str]
    resource_reports: List[ResourceIngestionReport] = []
    fatal_errors: List[dict] = []
    status: str = "running"

    @classmethod
    def start_new(cls, run_id: Optional[str] = None) -> "IngestionRunReport":
        return cls(run_id=run_id)

    def add_fatal_error(self, location: str, error: str) -> None:
        self.fatal_errors.append({"location": location, "error": error})

    def finalize(self) -> None:
        if self.fatal_errors:
            self.status = "completed_with_errors"
        else:
            self.status = "completed"
""",

    # loaders
    "radiolab_atlas/loaders/__init__.py": "",
    "radiolab_atlas/loaders/document_loader.py": r"""from typing import Tuple

from radiolab_atlas.models.ingestion import RawDocument, ResourceMetadata


class DocumentLoader:
    """Minimal loader: treat location as a UTF-8 text file on disk.

    Later we will extend this to handle PDF/DOCX/PPTX and remote sources.
    """

    def __init__(self, settings):
        self.settings = settings

    def load(self, location: str) -> Tuple[RawDocument, ResourceMetadata]:
        with open(location, "r", encoding="utf-8") as f:
            text = f.read()

        resource_id = location.replace("\\\\", "/").split("/")[-1]

        meta = ResourceMetadata(
            id=resource_id,
            title=resource_id,
            resource_type="Report",  # should match ontology controlled vocab
            file_type="TXT",
        )

        raw_doc = RawDocument(
            resource_id=resource_id,
            location=location,
            text=text,
            structure_hints=None,
        )

        return raw_doc, meta
""",

    # chunkers
    "radiolab_atlas/chunkers/__init__.py": "",
    "radiolab_atlas/chunkers/default_chunker.py": r"""from typing import List

from radiolab_atlas.models.ingestion import RawDocument, ResourceMetadata, TextChunk


class DefaultChunker:
    """Naive chunker: splits document into fixed-size character chunks.

    This is good enough for an MVP; later we can upgrade to LangChain text splitters.
    """

    def __init__(self, settings):
        self.settings = settings

    def chunk(self, raw_doc: RawDocument, resource_meta: ResourceMetadata) -> List[TextChunk]:
        text = raw_doc.text
        chunk_size = self.settings.chunk_size
        chunks: List[TextChunk] = []

        for i in range(0, len(text), chunk_size):
            chunk_text = text[i : i + chunk_size]
            chunk = TextChunk(
                chunk_id=f"{raw_doc.resource_id}_chunk_{len(chunks)}",
                resource_id=raw_doc.resource_id,
                text=chunk_text,
                order_index=len(chunks),
            )
            chunks.append(chunk)

        return chunks
""",

    # classifiers
    "radiolab_atlas/classifiers/__init__.py": "",
    "radiolab_atlas/classifiers/ontology_classifier.py": r"""from typing import List

from radiolab_atlas.models.ingestion import ResourceMetadata, TextChunk, ClassifiedChunk


class OntologyClassifier:
    """Stub classifier for Phase 1.

    For now, it assigns fixed ontology IDs to each chunk so we can
    test the pipeline end to end without calling an LLM.
    """

    def __init__(self, settings, ontology):
        self.settings = settings
        self.ontology = ontology

    def classify(
        self,
        chunks: List[TextChunk],
        resource_meta: ResourceMetadata,
    ) -> List[ClassifiedChunk]:
        classified: List[ClassifiedChunk] = []
        for chunk in chunks:
            classified.append(
                ClassifiedChunk(
                    chunk=chunk,
                    concept_ids=["concept:CRC_Operations"],
                    scenario_ids=["scenario:Urban_Fallout_CRC_Day2"],
                )
            )
        return classified
""",

    # validators
    "radiolab_atlas/validators/__init__.py": "",
    "radiolab_atlas/validators/ontology_validator.py": r"""from typing import List, Tuple

from radiolab_atlas.models.ingestion import ClassifiedChunk, ResourceMetadata
from radiolab_atlas.models.graph import (
    ValidatedGraphBatch,
    GraphNode,
    GraphRelationship,
)


class ValidationReport:
    def __init__(self):
        self.warnings: List[str] = []
        self.errors: List[str] = []


class OntologyValidator:
    """Stub validator for Phase 1.

    Wraps classified chunks into a simple graph and does not yet
    enforce the full ontology.json constraints.
    """

    def __init__(self, ontology):
        self.ontology = ontology

    def validate(
        self,
        classified_chunks: List[ClassifiedChunk],
        resource_meta: ResourceMetadata,
    ) -> Tuple[ValidatedGraphBatch, ValidationReport]:
        report = ValidationReport()

        nodes: List[GraphNode] = []
        relationships: List[GraphRelationship] = []

        # Resource node
        resource_node = GraphNode(
            id=resource_meta.id,
            label="Resource",
            properties=resource_meta.model_dump(exclude_none=True),
        )
        nodes.append(resource_node)

        concept_ids = set()
        scenario_ids = set()

        for c in classified_chunks:
            for cid in c.concept_ids:
                if cid not in concept_ids:
                    concept_ids.add(cid)
                    nodes.append(
                        GraphNode(
                            id=cid,
                            label="Concept",
                            properties={"id": cid},
                        )
                    )
                    relationships.append(
                        GraphRelationship(
                            id=f"{resource_meta.id}_TEACHES_{cid}",
                            type="TEACHES",
                            start_id=resource_meta.id,
                            end_id=cid,
                            properties={},
                        )
                    )
            for sid in c.scenario_ids:
                if sid not in scenario_ids:
                    scenario_ids.add(sid)
                    nodes.append(
                        GraphNode(
                            id=sid,
                            label="Scenario",
                            properties={"id": sid},
                        )
                    )
                    relationships.append(
                        GraphRelationship(
                            id=f"{resource_meta.id}_APPLIES_TO_{sid}",
                            type="APPLIES_TO",
                            start_id=resource_meta.id,
                            end_id=sid,
                            properties={},
                        )
                    )

        batch = ValidatedGraphBatch(nodes=nodes, relationships=relationships)
        return batch, report
""",

    # writers
    "radiolab_atlas/writers/__init__.py": "",
    "radiolab_atlas/writers/neo4j_writer.py": r"""from radiolab_atlas.models.graph import ValidatedGraphBatch
from radiolab_atlas.utils.logging_utils import get_logger

logger = get_logger(__name__)


class Neo4jWriter:
    """Stub writer: logs what would be written to Neo4j."""

    def __init__(self, neo4j_driver=None):
        self.driver = neo4j_driver

    def write(self, graph_batch: ValidatedGraphBatch) -> dict:
        logger.info(
            "Neo4jWriter: would write %d nodes and %d relationships",
            len(graph_batch.nodes),
            len(graph_batch.relationships),
        )
        return {
            "nodes": len(graph_batch.nodes),
            "relationships": len(graph_batch.relationships),
            "status": "stubbed",
        }
""",
    "radiolab_atlas/writers/postgres_writer.py": r"""from typing import List

from radiolab_atlas.models.ingestion import ResourceMetadata, ClassifiedChunk
from radiolab_atlas.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PostgresWriter:
    """Stub writer: logs what would be written to Postgres."""

    def __init__(self, engine=None):
        self.engine = engine

    def write(self, resource_meta: ResourceMetadata, chunks: List[ClassifiedChunk]) -> dict:
        logger.info(
            "PostgresWriter: would write resource=%s with %d chunks",
            resource_meta.id,
            len(chunks),
        )
        return {
            "resource_id": resource_meta.id,
            "chunks": len(chunks),
            "status": "stubbed",
        }
""",
    "radiolab_atlas/writers/vector_writer.py": r"""from typing import List

from radiolab_atlas.models.ingestion import ClassifiedChunk
from radiolab_atlas.utils.logging_utils import get_logger

logger = get_logger(__name__)


class VectorWriter:
    """Stub writer: logs what would be written to ChromaDB."""

    def __init__(self, chroma_client=None, embedding_fn=None):
        self.client = chroma_client
        self.embedding_fn = embedding_fn

    def write(self, chunks: List[ClassifiedChunk]) -> dict:
        logger.info(
            "VectorWriter: would embed and write %d chunks",
            len(chunks),
        )
        return {
            "chunks": len(chunks),
            "status": "stubbed",
        }
""",

    # pipelines / orchestrator
    "radiolab_atlas/pipelines/__init__.py": "",
    "radiolab_atlas/pipelines/orchestrator.py": r"""from __future__ import annotations

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
""",

    # CLI entrypoint
    "cli.py": r"""from radiolab_atlas.config import Settings
from radiolab_atlas.pipelines.orchestrator import (
    IngestionOrchestrator,
    OrchestratorDependencies,
)
from radiolab_atlas.loaders.document_loader import DocumentLoader
from radiolab_atlas.chunkers.default_chunker import DefaultChunker
from radiolab_atlas.classifiers.ontology_classifier import OntologyClassifier
from radiolab_atlas.validators.ontology_validator import OntologyValidator
from radiolab_atlas.writers.neo4j_writer import Neo4jWriter
from radiolab_atlas.writers.postgres_writer import PostgresWriter
from radiolab_atlas.writers.vector_writer import VectorWriter


def main():
    settings = Settings()

    # TODO: load ontology.json properly
    ontology = {}

    deps = OrchestratorDependencies(
        document_loader=DocumentLoader(settings),
        chunker=DefaultChunker(settings),
        classifier=OntologyClassifier(settings, ontology),
        validator=OntologyValidator(ontology),
        neo4j_writer=Neo4jWriter(),
        postgres_writer=PostgresWriter(),
        vector_writer=VectorWriter(),
    )

    orchestrator = IngestionOrchestrator(settings, deps)

    test_location = "data/input/test.txt"
    report = orchestrator.ingest_documents([test_location], run_id="demo-run-1")

    print(report.model_dump(indent=2))


if __name__ == "__main__":
    main()
""",

    # tests
    "tests/__init__.py": "",
    "tests/test_smoke_orchestrator.py": r"""from radiolab_atlas.config import Settings
from radiolab_atlas.pipelines.orchestrator import (
    IngestionOrchestrator,
    OrchestratorDependencies,
)
from radiolab_atlas.loaders.document_loader import DocumentLoader
from radiolab_atlas.chunkers.default_chunker import DefaultChunker
from radiolab_atlas.classifiers.ontology_classifier import OntologyClassifier
from radiolab_atlas.validators.ontology_validator import OntologyValidator
from radiolab_atlas.writers.neo4j_writer import Neo4jWriter
from radiolab_atlas.writers.postgres_writer import PostgresWriter
from radiolab_atlas.writers.vector_writer import VectorWriter


def test_smoke_ingestion(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a small test document.")

    settings = Settings()
    ontology = {}
    deps = OrchestratorDependencies(
        document_loader=DocumentLoader(settings),
        chunker=DefaultChunker(settings),
        classifier=OntologyClassifier(settings, ontology),
        validator=OntologyValidator(ontology),
        neo4j_writer=Neo4jWriter(),
        postgres_writer=PostgresWriter(),
        vector_writer=VectorWriter(),
    )

    orchestrator = IngestionOrchestrator(settings, deps)
    report = orchestrator.ingest_documents([str(test_file)], run_id="pytest-run")

    assert report.status in ("completed", "completed_with_errors")
    assert len(report.resource_reports) == 1
""",

    # sample input file
    "data/input/test.txt": "This is a test document for the Radiolab Atlas ingestion pipeline.\n"
}


def main() -> None:
    root = Path(".").resolve()
    for rel_path, content in FILES.items():
        path = root / rel_path
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            print(f"skipped existing file: {rel_path}")
            continue

        path.write_text(content, encoding="utf-8")
        print(f"created file: {rel_path}")


if __name__ == "__main__":
    main()
