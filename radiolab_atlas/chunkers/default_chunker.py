from typing import List

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
