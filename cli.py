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
from radiolab_atlas.utils.ontology_loader import load_ontology



def main():
    settings = Settings()

    ontology = load_ontology(settings.ontology_path)

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

    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
