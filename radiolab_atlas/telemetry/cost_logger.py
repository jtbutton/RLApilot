from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from radiolab_atlas.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class CostEvent:
    timestamp: float
    run_id: str
    doc_id: str
    event_type: str
    batch_id: Optional[str] = None
    chunk_ids: Optional[List[str]] = None

    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    cost_usd: Optional[float] = None

    estimated_tokens_in: Optional[int] = None
    estimated_tokens_out: Optional[int] = None
    estimated_cost_usd: Optional[float] = None

    metadata: Optional[Dict[str, Any]] = None


class CostLogger:
    def __init__(self, output_path: str = "data/logs/cost_events.jsonl"):
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: CostEvent) -> None:
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(event)) + "\n")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to write cost event. Error=%s", exc)

    def emit_prefilter_rollup(
        self,
        run_id: str,
        doc_id: str,
        counts_by_decision: Dict[str, int],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.emit(
            CostEvent(
                timestamp=time.time(),
                run_id=run_id,
                doc_id=doc_id,
                event_type="prefilter",
                metadata={"counts_by_decision": counts_by_decision, **(metadata or {})},
            )
        )

    def emit_doc_rollup(
        self,
        run_id: str,
        doc_id: str,
        llm_calls: int,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.emit(
            CostEvent(
                timestamp=time.time(),
                run_id=run_id,
                doc_id=doc_id,
                event_type="doc_rollup",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd,
                metadata=metadata,
            )
        )
