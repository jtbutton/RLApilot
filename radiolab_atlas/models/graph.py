from typing import Any, Dict, List
from pydantic import BaseModel


class GraphNode(BaseModel):
    id: str
    label: str
    properties: Dict[str, Any]


class GraphRelationship(BaseModel):
    id: str
    type: str
    start_id: str
    end_id: str
    properties: Dict[str, Any] = {}


class ValidatedGraphBatch(BaseModel):
    nodes: List[GraphNode]
    relationships: List[GraphRelationship]
