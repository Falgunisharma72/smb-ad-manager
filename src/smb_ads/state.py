"""Internal state of the simulated Meta Ads account.

The Env holds one of these. Everything the agent does flows through mutator
methods here, which makes it trivial to snapshot / serialize / replay.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .models import (
    Ad,
    AdSet,
    Campaign,
    Creative,
    Industry,
    Metrics,
    Objective,
    PolicyRule,
    SMBProfile,
)


@dataclass
class AgentCallLog:
    """Every action the agent takes is logged here — feeds anti-hack detectors."""
    step: int
    tool: str
    args: dict
    reasoning: str
    result_ok: bool
    error: Optional[str] = None


@dataclass
class AccountState:
    """Everything the mock Meta Marketing API mutates."""

    smb: SMBProfile
    total_budget_inr: float
    budget_spent_inr: float = 0.0

    campaigns: dict[str, Campaign] = field(default_factory=dict)
    ad_sets: dict[str, AdSet] = field(default_factory=dict)
    ads: dict[str, Ad] = field(default_factory=dict)

    # Daily metrics history — keyed by (campaign_id, day)
    metrics_history: dict[tuple[str, int], Metrics] = field(default_factory=dict)

    # Currently active policies (subset of full rulebook — adjusted by drift)
    active_policies: dict[str, PolicyRule] = field(default_factory=dict)

    # Agent call log — drives anti-hack detectors and format-compliance reward
    call_log: list[AgentCallLog] = field(default_factory=list)

    # Counters for quick anti-hack checks
    rejected_ads_this_episode: int = 0
    malformed_action_count: int = 0

    # Generators for deterministic IDs
    _next_id_counters: dict[str, int] = field(default_factory=lambda: {
        "campaign": 0, "ad_set": 0, "ad": 0,
    })

    def next_id(self, kind: str) -> str:
        """Generate deterministic sequential IDs like c001, as001, ad001."""
        prefix = {"campaign": "c", "ad_set": "as", "ad": "ad"}[kind]
        self._next_id_counters[kind] += 1
        return f"{prefix}{self._next_id_counters[kind]:03d}"

    @property
    def budget_remaining_inr(self) -> float:
        return self.total_budget_inr - self.budget_spent_inr

    @property
    def active_campaigns(self) -> list[Campaign]:
        return [c for c in self.campaigns.values() if c.status == "active"]

    @property
    def active_ad_sets(self) -> list[AdSet]:
        return [a for a in self.ad_sets.values() if a.status == "active"]

    @property
    def active_ads(self) -> list[Ad]:
        return [a for a in self.ads.values() if a.status == "active"]

    def paused_fraction(self) -> float:
        """Fraction of campaigns currently paused — input to anti-hack r5."""
        if not self.campaigns:
            return 0.0
        paused = sum(1 for c in self.campaigns.values() if c.status == "paused")
        return paused / len(self.campaigns)

    def most_recent_metrics(self, campaign_id: str) -> Optional[Metrics]:
        """Latest metrics for a campaign, if any."""
        keys = [k for k in self.metrics_history if k[0] == campaign_id]
        if not keys:
            return None
        latest_day = max(k[1] for k in keys)
        return self.metrics_history[(campaign_id, latest_day)]

    def to_dict(self) -> dict:
        """JSON-serializable snapshot — used by env.state()."""
        return {
            "smb": self.smb.model_dump(),
            "total_budget_inr": self.total_budget_inr,
            "budget_spent_inr": self.budget_spent_inr,
            "budget_remaining_inr": self.budget_remaining_inr,
            "campaigns": {k: v.model_dump() for k, v in self.campaigns.items()},
            "ad_sets": {k: v.model_dump() for k, v in self.ad_sets.items()},
            "ads": {k: v.model_dump() for k, v in self.ads.items()},
            "rejected_ads_this_episode": self.rejected_ads_this_episode,
            "malformed_action_count": self.malformed_action_count,
            "paused_fraction": self.paused_fraction(),
            "call_log_length": len(self.call_log),
        }
