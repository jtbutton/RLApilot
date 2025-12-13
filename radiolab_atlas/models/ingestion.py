from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel


class ResourceMetadata(BaseModel):
    """Minimal metadata for a Resource node, aligned with ontology fields."""

    id: str
    title: str
    resource_type: str
    summary: Optional[str] = None
    source_program: Optional[str] = None
    version: Optional[str] = None
    date_published: Optional[str] = None  # ISO string
    file_type: Optional[str] = None
    authors: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    ontology_version: Optional[str] = None


class RawDocument(BaseModel):
    resource_id: str
    location: str
    text: str
    structure_hints: Optional[dict] = None  # pages/slides/headings/etc.


class TextChunk(BaseModel):
    chunk_id: str
    resource_id: str
    text: str
    order_index: int
    span: Optional[dict] = None  # e.g. {\"pages\": [1, 2]} or {\"slide\": 3}


class ClassifiedChunk(BaseModel):
    chunk: TextChunk
    concept_ids: List[str] = []
    competency_ids: List[str] = []
    scenario_ids: List[str] = []
    role_ids: List[str] = []
    instrument_ids: List[str] = []
    network_program_ids: List[str] = []
    candidate_concept_labels: List[str] = []
