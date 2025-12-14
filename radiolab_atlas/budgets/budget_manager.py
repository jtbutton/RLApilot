from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from radiolab_atlas.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BudgetDecision(str, Enum):
    ALLOW = "ALLOW"
    THROTTLE = "THROTTLE"
    DOWNGRADE = "DOWNGRADE"


@dataclass
class BudgetPolicy:
    max_cost_per_run_usd: float = 2.00
    max_cost_per_doc_usd: float = 0.50
    max_cost_per_day_usd: float = 5.00

    max_calls_per_minute: int = 30
    throttle_seconds: float = 2.0

    downgrade_on_quota_error: bool = True
    dry_run: bool = False


@dataclass
class UsageTotals:
    run_cost_usd: float = 0.0
    doc_cost_usd: float = 0.0
    day_cost_usd: float = 0.0
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0


class TokenEstimator:
    def estimate_tokens(self, text: str, model: str) -> int:
        return max(1, int(len(text) / 4))


class BudgetManager:
    def __init__(self, policy: BudgetPolicy, estimator: Optional[TokenEstimator] = None):
        self.policy = policy
        self.estimator = estimator or TokenEstimator()
        self.totals = UsageTotals()
        self._call_timestamps: List[float] = []
        self._llm_disabled: bool = False

    def llm_disabled(self) -> bool:
        return self._llm_disabled

    def disable_llm(self) -> None:
        self._llm_disabled = True

    def preflight_batch(
        self,
        model: str,
        prompt_text: str,
        expected_output_tokens: int,
        doc_id: str,
        run_id: str,
        price_table: Optional[Dict[str, Any]] = None,
    ) -> Tuple[BudgetDecision, Dict[str, Any]]:
        if self._llm_disabled:
            return BudgetDecision.DOWNGRADE, {"reason": "LLM_DISABLED"}

        decision = self._rate_limit_decision()
        if decision != BudgetDecision.ALLOW:
            return decision, {"reason": "RATE_LIMIT"}

        est_in = self.estimator.estimate_tokens(prompt_text, model=model)
        est_out = int(expected_output_tokens)
        est_cost = self._estimate_cost_usd(model, est_in, est_out, price_table=price_table)

        telemetry = {
            "estimated_tokens_in": est_in,
            "estimated_tokens_out": est_out,
            "estimated_cost_usd": est_cost,
            "doc_id": doc_id,
            "run_id": run_id,
        }

        if self.totals.run_cost_usd + est_cost > self.policy.max_cost_per_run_usd:
            return BudgetDecision.DOWNGRADE, {**telemetry, "reason": "RUN_BUDGET_EXCEEDED"}
        if self.totals.doc_cost_usd + est_cost > self.policy.max_cost_per_doc_usd:
            return BudgetDecision.DOWNGRADE, {**telemetry, "reason": "DOC_BUDGET_EXCEEDED"}
        if self.totals.day_cost_usd + est_cost > self.policy.max_cost_per_day_usd:
            return BudgetDecision.DOWNGRADE, {**telemetry, "reason": "DAY_BUDGET_EXCEEDED"}

        if self.policy.dry_run:
            return BudgetDecision.DOWNGRADE, {**telemetry, "reason": "DRY_RUN_ESTIMATE_ONLY"}

        return BudgetDecision.ALLOW, telemetry

    def record_actual(
        self,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        doc_id: str,
        run_id: str,
    ) -> None:
        self.totals.llm_calls += 1
        self.totals.tokens_in += tokens_in
        self.totals.tokens_out += tokens_out
        self.totals.run_cost_usd += cost_usd
        self.totals.doc_cost_usd += cost_usd
        self.totals.day_cost_usd += cost_usd
        self._call_timestamps.append(time.time())

    def handle_quota_or_rate_error(self, exc: Exception) -> BudgetDecision:
        msg = str(exc).lower()
        if self.policy.downgrade_on_quota_error and ("insufficient_quota" in msg or "exceeded your current quota" in msg):
            self.disable_llm()
            return BudgetDecision.DOWNGRADE
        return BudgetDecision.THROTTLE

    def _rate_limit_decision(self) -> BudgetDecision:
        now = time.time()
        window = [t for t in self._call_timestamps if now - t < 60.0]
        self._call_timestamps = window

        if len(window) >= self.policy.max_calls_per_minute:
            time.sleep(self.policy.throttle_seconds)
            return BudgetDecision.THROTTLE
        return BudgetDecision.ALLOW

    @staticmethod
    def _estimate_cost_usd(
        model: str,
        tokens_in: int,
        tokens_out: int,
        price_table: Optional[Dict[str, Any]] = None,
    ) -> float:
        if not price_table or model not in price_table:
            return 0.0

        p = price_table[model]
        in_cost = (tokens_in / 1000.0) * float(p["input_per_1k"])
        out_cost = (tokens_out / 1000.0) * float(p["output_per_1k"])
        return in_cost + out_cost
