from typing import List

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
