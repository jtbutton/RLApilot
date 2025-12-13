from typing import List

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
