from radiolab_atlas.models.graph import ValidatedGraphBatch
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
