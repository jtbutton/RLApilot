from typing import List, Tuple

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
