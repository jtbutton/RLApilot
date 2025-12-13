from typing import List, Tuple

from radiolab_atlas.models.ingestion import ClassifiedChunk, ResourceMetadata
from radiolab_atlas.models.graph import (
    ValidatedGraphBatch,
    GraphNode,
    GraphRelationship,
)
from radiolab_atlas.utils.ontology_loader import Ontology


class ValidationReport:
    """
    Collects warnings and errors discovered during validation.
    """

    def __init__(self) -> None:
        self.warnings: List[str] = []
        self.errors: List[str] = []


class OntologyValidator:
    """
    Ontology-aware validator (VA-1):

    - Checks that node labels exist in ontology.node_types.
    - Checks that relationship types exist in ontology.relationship_types.
    - Validates and normalizes controlled vocabularies like resource_type.

    This is still "v0.1": it does not validate instance IDs of concepts/scenarios,
    only the types and vocab values.
    """

    def __init__(self, ontology: Ontology):
        self.ontology = ontology
        self.valid_node_labels = ontology.node_labels
        self.valid_rel_types = ontology.relationship_types
        self.resource_type_values = ontology.get_vocab_values("resource_types")

    def validate(
        self,
        classified_chunks: List[ClassifiedChunk],
        resource_meta: ResourceMetadata,
    ) -> Tuple[ValidatedGraphBatch, ValidationReport]:
        report = ValidationReport()

        # 1. Normalize / validate resource_type against controlled vocab.
        normalized_meta = self._validate_resource_metadata(resource_meta, report)

        nodes: List[GraphNode] = []
        relationships: List[GraphRelationship] = []

        # 2. Resource node
        resource_label = "Resource"
        self._assert_valid_node_label(resource_label, report)

        resource_node = GraphNode(
            id=normalized_meta.id,
            label=resource_label,
            properties=normalized_meta.model_dump(exclude_none=True),
        )
        nodes.append(resource_node)

        # 3. Build Concept + Scenario nodes and TEACHES / APPLIES_TO relationships
        concept_ids = set()
        scenario_ids = set()

        for c in classified_chunks:
            # Concepts
            for cid in c.concept_ids:
                if cid not in concept_ids:
                    concept_ids.add(cid)
                    concept_label = "Concept"
                    self._assert_valid_node_label(concept_label, report)

                    nodes.append(
                        GraphNode(
                            id=cid,
                            label=concept_label,
                            properties={"id": cid},
                        )
                    )

                    rel_type = "TEACHES"
                    self._assert_valid_rel_type(rel_type, report)

                    relationships.append(
                        GraphRelationship(
                            id=f"{normalized_meta.id}_{rel_type}_{cid}",
                            type=rel_type,
                            start_id=normalized_meta.id,
                            end_id=cid,
                            properties={},
                        )
                    )

            # Scenarios
            for sid in c.scenario_ids:
                if sid not in scenario_ids:
                    scenario_ids.add(sid)
                    scenario_label = "Scenario"
                    self._assert_valid_node_label(scenario_label, report)

                    nodes.append(
                        GraphNode(
                            id=sid,
                            label=scenario_label,
                            properties={"id": sid},
                        )
                    )

                    rel_type = "APPLIES_TO"
                    self._assert_valid_rel_type(rel_type, report)

                    relationships.append(
                        GraphRelationship(
                            id=f"{normalized_meta.id}_{rel_type}_{sid}",
                            type=rel_type,
                            start_id=normalized_meta.id,
                            end_id=sid,
                            properties={},
                        )
                    )

        batch = ValidatedGraphBatch(nodes=nodes, relationships=relationships)
        return batch, report

    # ---- helpers ----

    def _validate_resource_metadata(
        self, resource_meta: ResourceMetadata, report: ValidationReport
    ) -> ResourceMetadata:
        """
        Validate and normalize ResourceMetadata fields against ontology vocabularies.
        For now we only check resource_type.
        """
        if self.resource_type_values is None or resource_meta.resource_type is None:
            return resource_meta

        if resource_meta.resource_type not in self.resource_type_values:
            report.warnings.append(
                f"resource_type '{resource_meta.resource_type}' not in ontology "
                f"resource_types; coercing to 'Other'."
            )
            resource_meta = resource_meta.model_copy(
                update={"resource_type": "Other"}
            )

        return resource_meta

    def _assert_valid_node_label(self, label: str, report: ValidationReport) -> None:
        if label not in self.valid_node_labels:
            report.errors.append(
                f"Node label '{label}' not found in ontology node_types."
            )

    def _assert_valid_rel_type(self, rel_type: str, report: ValidationReport) -> None:
        if rel_type not in self.valid_rel_types:
            report.errors.append(
                f"Relationship type '{rel_type}' not found in ontology relationship_types."
            )
