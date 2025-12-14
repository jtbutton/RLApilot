from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from neo4j import GraphDatabase

from radiolab_atlas.models.graph import GraphNode, GraphRelationship, ValidatedGraphBatch
from radiolab_atlas.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Neo4jWriteSummary:
    nodes_merged: int
    relationships_merged: int
    constraints_created: int
    status: str


class Neo4jWriter:
    """
    Production MERGE-based Neo4j writer (skeleton).
    """

    def __init__(self, uri: str, user: str, password: str, database: Optional[str] = None):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self) -> None:
        self.driver.close()

    def ensure_constraints(self) -> int:
        constraints_created = 0
        statements = [
            "CREATE CONSTRAINT resource_id IF NOT EXISTS FOR (n:Resource) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (n:Concept) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT scenario_id IF NOT EXISTS FOR (n:Scenario) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT competency_id IF NOT EXISTS FOR (n:Competency) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT role_id IF NOT EXISTS FOR (n:Role) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT instrument_id IF NOT EXISTS FOR (n:Instrument) REQUIRE n.id IS UNIQUE",
        ]
        try:
            with self.driver.session(database=self.database) as session:
                for stmt in statements:
                    session.run(stmt)
                    constraints_created += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Neo4j constraint creation failed or not permitted. Error=%s", exc)
        return constraints_created

    def write(self, batch: ValidatedGraphBatch, provenance: Dict[str, Any]) -> Neo4jWriteSummary:
        constraints_created = self.ensure_constraints()
        nodes_merged = 0
        rels_merged = 0
        ingested_at = provenance.get("ingested_at") or time.time()

        try:
            with self.driver.session(database=self.database) as session:
                for node in batch.nodes:
                    self._merge_node(session, node, provenance, ingested_at)
                    nodes_merged += 1

                for rel in batch.relationships:
                    self._merge_relationship(session, rel, provenance, ingested_at)
                    rels_merged += 1

            return Neo4jWriteSummary(
                nodes_merged=nodes_merged,
                relationships_merged=rels_merged,
                constraints_created=constraints_created,
                status="ok",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Neo4j write failed")
            return Neo4jWriteSummary(
                nodes_merged=nodes_merged,
                relationships_merged=rels_merged,
                constraints_created=constraints_created,
                status=f"error: {exc}",
            )

    @staticmethod
    def _merge_node(session, node: GraphNode, provenance: Dict[str, Any], ingested_at: float) -> None:
        label = node.label
        props = dict(node.properties or {})
        props.setdefault("id", node.id)

        cypher = f"""
        MERGE (n:`{label}` {{ id: $id }})
        SET n += $props
        SET n.source_doc = $source_doc,
            n.run_id = $run_id,
            n.ingested_at = $ingested_at
        """
        session.run(
            cypher,
            id=node.id,
            props=props,
            source_doc=provenance.get("source_doc"),
            run_id=provenance.get("run_id"),
            ingested_at=ingested_at,
        )

    @staticmethod
    def _merge_relationship(session, rel: GraphRelationship, provenance: Dict[str, Any], ingested_at: float) -> None:
        rel_type = rel.type
        props = dict(rel.properties or {})

        cypher = f"""
        MATCH (a {{ id: $start_id }})
        MATCH (b {{ id: $end_id }})
        MERGE (a)-[r:`{rel_type}`]->(b)
        SET r += $props
        SET r.source_doc = $source_doc,
            r.run_id = $run_id,
            r.ingested_at = $ingested_at
        """
        session.run(
            cypher,
            start_id=rel.start_id,
            end_id=rel.end_id,
            props=props,
            source_doc=provenance.get("source_doc"),
            run_id=provenance.get("run_id"),
            ingested_at=ingested_at,
        )
