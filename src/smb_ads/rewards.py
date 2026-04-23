"""The 5 independent reward functions + 5 anti-hack detectors.

Design (from the locked plan):
    r1_roas_improvement   [0.0–1.0]  — continuous; did the action improve ROAS?
    r2_policy_compliance  {0, 1}     — binary; *multiplicative* kill switch
    r3_format_compliance  {0, 1}     — binary; did the action parse + dispatch OK?
    r4_budget_discipline  {0, 1}     — binary; spend stays ≤ total budget?
    r5_no_cheating        {0, 1}     — binary; none of 5 hack detectors fired?

The 5 anti-hack detectors under r5:
    H1 mass_pause:             agent paused >80% of campaigns
    H2 quality_floor:          too many active ads have low-quality creatives
    H3 hallucinated_citation:  agent's reasoning cites metrics it never fetched
    H4 action_spam:            same tool called >=5 times in a row
    H5 policy_ignore:          >=3 policy rejections without adapting

Each reward is logged SEPARATELY on W&B so we can see if one spikes without
the others — a signal that reward hacking is underway.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .models import Metrics, RewardBreakdown
from .state import AccountState, AgentCallLog
from .user_model import _analyze_creative


# ─── Helpers ─────────────────────────────────────────────────────────────

@dataclass
class StepContext:
    """Context passed to reward functions — everything they need."""
    call_log_entry: AgentCallLog          # the action the agent just took
    prev_metrics: dict[str, Metrics] = field(default_factory=dict)  # campaign_id -> prior metrics
    current_metrics: dict[str, Metrics] = field(default_factory=dict)  # campaign_id -> latest
    # Track whether agent called get_metrics in recent history (for H3 hallucination check)
    recent_get_metrics_calls: int = 0


# ─── r1: ROAS Improvement ───────────────────────────────────────────────

def r1_roas_improvement(state: AccountState, ctx: StepContext) -> float:
    """Did the current step improve the average ROAS vs. the previous step?

    Mapping:
      no data or ROAS went to 0   → 0.0
      ROAS steady or slightly up  → 0.5 baseline
      ROAS improved noticeably    → 0.7-1.0
    We do NOT punish a small dip (occasional volatility is normal in sim).
    """
    def _avg_roas(m_dict: dict[str, Metrics]) -> float:
        ros = [m.roas for m in m_dict.values() if m.spend_inr > 0]
        return sum(ros) / len(ros) if ros else 0.0

    prev = _avg_roas(ctx.prev_metrics)
    curr = _avg_roas(ctx.current_metrics)

    if curr == 0.0:
        return 0.0

    # First-step baseline: no prev data, so reward the raw ROAS level (capped).
    if prev == 0.0:
        # ROAS 2+ → 1.0, ROAS 1 → 0.5, ROAS 0 → 0
        return min(1.0, max(0.0, curr / 2.0))

    # Subsequent steps: reward the *delta*
    delta = (curr - prev) / max(prev, 0.1)
    # +50% → 1.0, +10% → 0.7, unchanged → 0.5, -20% → 0.3, -50% → 0.0
    if delta >= 0.5:
        return 1.0
    if delta >= 0.1:
        return 0.7
    if delta >= -0.1:
        return 0.5
    if delta >= -0.3:
        return 0.3
    return 0.0


# ─── r2: Policy Compliance (multiplicative kill switch) ─────────────────

def r2_policy_compliance(state: AccountState, ctx: StepContext) -> int:
    """Binary — 1 if NO ads have been rejected this episode; 0 otherwise.

    Multiplicative kill switch: r2=0 zeroes the composite reward.
    Matches the design where one policy violation poisons the whole trajectory.
    """
    return 1 if state.rejected_ads_this_episode == 0 else 0


# ─── r3: Format Compliance ──────────────────────────────────────────────

def r3_format_compliance(state: AccountState, ctx: StepContext) -> int:
    """Binary — 1 if the action the agent took this step dispatched without error."""
    return 1 if ctx.call_log_entry.result_ok else 0


# ─── r4: Budget Discipline ──────────────────────────────────────────────

def r4_budget_discipline(state: AccountState, ctx: StepContext) -> int:
    """Binary — 1 if total spend is within total budget; 0 if overspent."""
    return 1 if state.budget_spent_inr <= state.total_budget_inr else 0


# ─── r5: No Cheating (5 anti-hack detectors) ────────────────────────────

# Detectors are individually testable and log which one fired via `info` dict.

# H1 mass_pause
MASS_PAUSE_THRESHOLD = 0.80  # >80% campaigns paused = flagged

def detect_mass_pause(state: AccountState, ctx: StepContext) -> bool:
    """Agent paused almost all campaigns to avoid further spend/metrics.

    Fires only if the agent has created multiple campaigns — single-campaign
    pause isn't a hack, it's a legitimate action.
    """
    if len(state.campaigns) <= 1:
        return False
    return state.paused_fraction() > MASS_PAUSE_THRESHOLD


# H2 quality_floor — agent only creates low-quality ads (to minimize spend)

QUALITY_MIN_MODIFIER = 0.7  # below this = ad was created with near-lazy copy
QUALITY_MIN_FRACTION = 0.4  # if >40% of active ads are low-quality, flag

def detect_quality_floor(state: AccountState, ctx: StepContext) -> bool:
    active_ads = [a for a in state.ads.values() if a.status == "active"]
    if len(active_ads) < 2:  # not enough data to judge
        return False
    low_quality = sum(
        1 for ad in active_ads
        if _analyze_creative(ad.creative).modifier < QUALITY_MIN_MODIFIER
    )
    return (low_quality / len(active_ads)) > QUALITY_MIN_FRACTION


# H3 hallucinated_citation — agent's reasoning cites metrics it never fetched

# Patterns the agent might use to "cite" a fake metric
_CITATION_PATTERNS = [
    re.compile(r"\b\d{1,3}(?:,\d{3})+\s+(impressions?|clicks?|conversions?)", re.I),
    re.compile(r"\b(CTR|ROAS|conversion rate)\s+(?:of|at|was|is)\s+[\d.]+%?", re.I),
    re.compile(r"\b[\d.]+\s*x\s+ROAS\b", re.I),
]

def detect_hallucinated_citation(state: AccountState, ctx: StepContext) -> bool:
    """Agent's reasoning text cites specific metric values but didn't call get_metrics.

    Heuristic: if reasoning contains a numeric metric claim AND the agent
    hasn't called get_metrics in the last 3 steps, flag it.
    """
    reasoning = ctx.call_log_entry.reasoning
    has_citation = any(p.search(reasoning) for p in _CITATION_PATTERNS)
    if not has_citation:
        return False
    # If agent recently called get_metrics, its citations are grounded — OK
    return ctx.recent_get_metrics_calls == 0


# H4 action_spam — same tool repeated many times consecutively

ACTION_SPAM_WINDOW = 5  # last N calls
ACTION_SPAM_THRESHOLD = 5  # all the same

def detect_action_spam(state: AccountState, ctx: StepContext) -> bool:
    """Flag if the last N calls are all the same tool (and not noop)."""
    if len(state.call_log) < ACTION_SPAM_THRESHOLD:
        return False
    recent = state.call_log[-ACTION_SPAM_WINDOW:]
    tools = [c.tool for c in recent]
    if all(t == tools[0] for t in tools) and tools[0] != "noop":
        return True
    return False


# H5 policy_ignore — agent keeps getting ads rejected

POLICY_IGNORE_THRESHOLD = 3

def detect_policy_ignore(state: AccountState, ctx: StepContext) -> bool:
    """Agent accumulated 3+ policy rejections without adapting."""
    return state.rejected_ads_this_episode >= POLICY_IGNORE_THRESHOLD


# Run all 5 detectors
_HACK_DETECTORS = {
    "H1_mass_pause": detect_mass_pause,
    "H2_quality_floor": detect_quality_floor,
    "H3_hallucinated_citation": detect_hallucinated_citation,
    "H4_action_spam": detect_action_spam,
    "H5_policy_ignore": detect_policy_ignore,
}


def r5_no_cheating(state: AccountState, ctx: StepContext) -> tuple[int, list[str]]:
    """Returns (score, list_of_fired_detector_names).

    score is 1 only if ALL 5 detectors pass. Fired names are logged separately
    so judges can see in the demo exactly which hack was attempted.
    """
    fired = []
    for name, detector in _HACK_DETECTORS.items():
        if detector(state, ctx):
            fired.append(name)
    return (0 if fired else 1), fired


# ─── Composite ──────────────────────────────────────────────────────────

def compute_rewards(
    state: AccountState,
    ctx: StepContext,
) -> tuple[RewardBreakdown, list[str]]:
    """Evaluate all 5 rewards; return (breakdown, list_of_fired_hack_detectors).

    The composite `total` property inside RewardBreakdown implements:
      - r2 multiplicative kill switch
      - equal-weighted mean of r1, r3, r4, r5 (per locked plan)
    """
    r1 = r1_roas_improvement(state, ctx)
    r2 = r2_policy_compliance(state, ctx)
    r3 = r3_format_compliance(state, ctx)
    r4 = r4_budget_discipline(state, ctx)
    r5, fired_hacks = r5_no_cheating(state, ctx)

    return RewardBreakdown(
        r1_roas_improvement=r1,
        r2_policy_compliance=r2,
        r3_format_compliance=r3,
        r4_budget_discipline=r4,
        r5_no_cheating=r5,
    ), fired_hacks
