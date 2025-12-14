from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from radiolab_atlas.classifiers.ontology_classifier import OntologyClassifier
from radiolab_atlas.models.ingestion import ClassifiedChunk, ResourceMetadata, TextChunk
from radiolab_atlas.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class BatchConfig:
    batch_size: int = 6
    max_batch_chars: int = 30000  # safety cap to avoid huge prompts
    require_chunk_id_keyed_output: bool = True


class BatchClassifier:
    """
    Wraps existing OntologyClassifier without changing it.
    Batches multiple chunks into one LLM call when classifier is in LLM mode.

    In stub mode, delegates to base_classifier.classify().
    """

    def __init__(self, base_classifier: OntologyClassifier, config: BatchConfig):
        self.base = base_classifier
        self.config = config

    def make_batches(self, chunks: List[TextChunk]) -> List[List[TextChunk]]:
        batches: List[List[TextChunk]] = []
        cur: List[TextChunk] = []
        cur_chars = 0

        for ch in chunks:
            ch_len = len(ch.text or "")
            if cur and (len(cur) >= self.config.batch_size or cur_chars + ch_len > self.config.max_batch_chars):
                batches.append(cur)
                cur = []
                cur_chars = 0

            cur.append(ch)
            cur_chars += ch_len

        if cur:
            batches.append(cur)

        return batches

    def classify_batches(self, chunks: List[TextChunk], resource_meta: ResourceMetadata) -> List[ClassifiedChunk]:
        mode = getattr(self.base, "mode", "stub")
        if str(mode).lower() == "stub":
            return self.base.classify(chunks, resource_meta)

        output_by_id: Dict[str, ClassifiedChunk] = {}

        for batch_i, batch in enumerate(self.make_batches(chunks)):
            try:
                raw = self._classify_batch_llm(batch, resource_meta, batch_i=batch_i)
                mapped = self._map_batch_json_to_results(batch, raw)
                for cc in mapped:
                    output_by_id[cc.chunk.chunk_id] = cc
            except Exception as exc:  # noqa: BLE001
                logger.warning("Batch LLM classify failed (batch=%s). Falling back to empty classifications. Error=%s", batch_i, exc)
                for ch in batch:
                    output_by_id[ch.chunk_id] = self._empty_classification(ch)

        return [output_by_id.get(ch.chunk_id) or self._empty_classification(ch) for ch in chunks]

    def _classify_batch_llm(self, batch: List[TextChunk], resource_meta: ResourceMetadata, batch_i: int) -> Dict[str, Any]:
        llm_client = getattr(self.base, "llm_client", None)
        if llm_client is None:
            raise RuntimeError("Base classifier has no llm_client; cannot batch classify in LLM mode.")

        system_prompt = self._build_system_prompt_for_batch()
        user_prompt = self._build_user_prompt_for_batch(batch, resource_meta)

        raw_text = llm_client.classify_chunk(system_prompt=system_prompt, user_prompt=user_prompt)
        data = llm_client.parse_json_response(raw_text)
        return data

    def _build_system_prompt_for_batch(self) -> str:
        rubric_text = getattr(self.base, "rubric_text", "")
        return (
            "You classify multiple text chunks into an existing ontology. "
            "Return JSON only. No prose.\n\n"
            "Rubric:\n"
            f"{rubric_text}\n\n"
            "Return JSON with this schema:\n"
            "{\n"
            '  "results": {\n'
            '    "<chunk_id>": {\n'
            '      "concept_ids": [string],\n'
            '      "competency_ids": [string],\n'
            '      "scenario_ids": [string],\n'
            '      "role_ids": [string],\n'
            '      "instrument_ids": [string],\n'
            '      "network_program_ids": [string],\n'
            '      "candidate_concept_labels": [string]\n'
            "    }\n"
            "  }\n"
            "}\n"
        )

    def _build_user_prompt_for_batch(self, batch: List[TextChunk], resource_meta: ResourceMetadata) -> str:
        vocab_snapshot = getattr(self.base, "vocab_snapshot", {})
        vocab_json = json.dumps(vocab_snapshot, indent=2)

        payload = {
            "resource_id": resource_meta.id,
            "title": resource_meta.title,
            "ontology_vocabulary_snapshot": vocab_snapshot,
            "chunks": [
                {"chunk_id": ch.chunk_id, "span": ch.span, "text": ch.text}
                for ch in batch
            ],
        }

        return (
            "Use the following ontology vocabulary snapshot (valid IDs):\n"
            f"{vocab_json}\n\n"
            "Classify each chunk. Output must be keyed by chunk_id.\n\n"
            f"INPUT:\n{json.dumps(payload, indent=2)}"
        )

    def _map_batch_json_to_results(self, batch: List[TextChunk], data: Dict[str, Any]) -> List[ClassifiedChunk]:
        results = data.get("results")
        if not isinstance(results, dict):
            raise ValueError("Batch response missing 'results' dict.")

        out: List[ClassifiedChunk] = []
        by_id = {c.chunk_id: c for c in batch}

        for chunk_id, payload in results.items():
            if chunk_id not in by_id:
                continue
            if not isinstance(payload, dict):
                out.append(self._empty_classification(by_id[chunk_id]))
                continue
            out.append(self._classified_from_payload(by_id[chunk_id], payload))

        for ch in batch:
            if not any(cc.chunk.chunk_id == ch.chunk_id for cc in out):
                out.append(self._empty_classification(ch))

        return out

    @staticmethod
    def _list_of_str(payload: Dict[str, Any], key: str) -> List[str]:
        v = payload.get(key, [])
        if not isinstance(v, list):
            return []
        return [str(x) for x in v]

    def _classified_from_payload(self, chunk: TextChunk, payload: Dict[str, Any]) -> ClassifiedChunk:
        return ClassifiedChunk(
            chunk=chunk,
            concept_ids=self._list_of_str(payload, "concept_ids"),
            competency_ids=self._list_of_str(payload, "competency_ids"),
            scenario_ids=self._list_of_str(payload, "scenario_ids"),
            role_ids=self._list_of_str(payload, "role_ids"),
            instrument_ids=self._list_of_str(payload, "instrument_ids"),
            network_program_ids=self._list_of_str(payload, "network_program_ids"),
            candidate_concept_labels=self._list_of_str(payload, "candidate_concept_labels"),
        )

    def _empty_classification(self, chunk: TextChunk) -> ClassifiedChunk:
        return ClassifiedChunk(
            chunk=chunk,
            concept_ids=[],
            competency_ids=[],
            scenario_ids=[],
            role_ids=[],
            instrument_ids=[],
            network_program_ids=[],
            candidate_concept_labels=[],
        )
