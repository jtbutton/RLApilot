from typing import Tuple

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
