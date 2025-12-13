from typing import List

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
