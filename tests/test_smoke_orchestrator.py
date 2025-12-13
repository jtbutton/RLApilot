from radiolab_atlas.config import Settings
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
