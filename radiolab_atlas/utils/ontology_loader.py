from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set


class Ontology:
    """
    Thin wrapper around ontology.json that exposes convenient lookups:
    - valid node labels
    - valid relationship types
    - controlled vocabularies (e.g., resource_types)
    """

    def __init__(self, raw: Dict[str, Any]):
        self.raw = raw

        self.node_labels: Set[str] = set(raw.get("node_types", {}).keys())
        self.relationship_types: Set[str] = set(raw.get("relationship_types", {}).keys())
        self.controlled_vocabularies: Dict[str, Any] = raw.get(
            "controlled_vocabularies", {}
        )

    def is_valid_node_label(self, label: str) -> bool:
        return label in self.node_labels

    def is_valid_relationship_type(self, rel_type: str) -> bool:
        return rel_type in self.relationship_types

    def get_vocab_values(self, name: str) -> Optional[Set[str]]:
        """
        Return the set of allowed values for a given controlled vocabulary
        (e.g., "resource_types", "scenario_types", "instrument_types").
        """
        vocab = self.controlled_vocabularies.get(name)
        if vocab is None:
            return None

        # Most vocabularies are plain lists; some (like concept_categories) are dicts.
        if isinstance(vocab, list):
            return set(vocab)
        if isinstance(vocab, dict):
            # Flatten dict keys + values if values are lists; otherwise just keys.
            values: Set[str] = set(vocab.keys())
            for v in vocab.values():
                if isinstance(v, Iterable) and not isinstance(v, (str, bytes)):
                    values.update(str(x) for x in v)
                else:
                    values.add(str(v))
            return values
        return None


def load_ontology(path: str) -> Ontology:
    """
    Load ontology.json from the given path and return an Ontology object.
    """
    ontology_path = Path(path)
    if not ontology_path.exists():
        raise FileNotFoundError(f"ontology file not found at: {ontology_path}")

    with ontology_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    return Ontology(raw)
