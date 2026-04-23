"""Tests for the 5 reward functions + 5 anti-hack detectors."""
from __future__ import annotations

import pytest

from src.smb_ads import rewards
from src.smb_ads.models import (
    Ad,
    Campaign,
    Creative,
    Industry,
    Metrics,
    Objective,
    SMBProfile,
)
from src.smb_ads.rewards import (
    StepContext,
    compute_rewards,
    detect_action_spam,
    detect_hallucinated_citation,
    detect_mass_pause,
    detect_policy_ignore,
    detect_quality_floor,
    r1_roas_improvement,
    r2_policy_compliance,
    r3_format_compliance,
    r4_budget_discipline,
    r5_no_cheating,
)
from src.smb_ads.state import AccountState, AgentCallLog


# ─── Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def base_state():
    smb = SMBProfile(
        name="Test Biz",
        industry=Industry.SKINCARE,
        location="Mumbai",
        monthly_budget_inr=10000.0,
        goal="online_sales",
        description="test biz",
    )
    return AccountState(smb=smb, total_budget_inr=10000.0)


@pytest.fixture
def ok_call_log_entry():
    return AgentCallLog(
        step=1, tool="noop", args={}, reasoning="thinking",
        result_ok=True, error=None,
    )


@pytest.fixture
def failed_call_log_entry():
    return AgentCallLog(
        step=1, tool="create_campaign", args={},
        reasoning="bad action", result_ok=False, error="missing args",
    )


def _m(imp: int, clicks: int, conv: int, spend: float, rev: float) -> Metrics:
    return Metrics(
        impressions=imp, clicks=clicks, conversions=conv,
        spend_inr=spend, revenue_inr=rev,
    )


# ─── r1: ROAS improvement ────────────────────────────────────────────────

def test_r1_zero_when_no_metrics(base_state, ok_call_log_entry):
    ctx = StepContext(call_log_entry=ok_call_log_entry)
    assert r1_roas_improvement(base_state, ctx) == 0.0


def test_r1_baseline_when_first_step(base_state, ok_call_log_entry):
    """No prev metrics, just current ROAS = 2.5x → r1 = 1.0 (capped)."""
    ctx = StepContext(
        call_log_entry=ok_call_log_entry,
        prev_metrics={},
        current_metrics={"c1": _m(1000, 25, 3, 500.0, 1500.0)},  # roas = 3.0
    )
    assert r1_roas_improvement(base_state, ctx) == 1.0


def test_r1_rewards_improvement(base_state, ok_call_log_entry):
    """ROAS went from 1.0 → 2.0 (+100%) → reward 1.0."""
    ctx = StepContext(
        call_log_entry=ok_call_log_entry,
        prev_metrics={"c1": _m(1000, 20, 2, 500.0, 500.0)},    # roas = 1.0
        current_metrics={"c1": _m(1000, 20, 4, 500.0, 1000.0)}, # roas = 2.0
    )
    assert r1_roas_improvement(base_state, ctx) == 1.0


def test_r1_penalizes_decline(base_state, ok_call_log_entry):
    """ROAS went from 2.0 → 0.8 (-60%) → reward 0.0."""
    ctx = StepContext(
        call_log_entry=ok_call_log_entry,
        prev_metrics={"c1": _m(1000, 20, 4, 500.0, 1000.0)},
        current_metrics={"c1": _m(1000, 8, 1, 500.0, 400.0)},
    )
    assert r1_roas_improvement(base_state, ctx) == 0.0


def test_r1_neutral_when_stable(base_state, ok_call_log_entry):
    ctx = StepContext(
        call_log_entry=ok_call_log_entry,
        prev_metrics={"c1": _m(1000, 20, 3, 500.0, 1000.0)},   # roas 2.0
        current_metrics={"c1": _m(1000, 20, 3, 500.0, 1010.0)}, # roas 2.02
    )
    assert r1_roas_improvement(base_state, ctx) == 0.5


# ─── r2: Policy compliance ───────────────────────────────────────────────

def test_r2_passes_when_no_rejections(base_state, ok_call_log_entry):
    ctx = StepContext(call_log_entry=ok_call_log_entry)
    assert r2_policy_compliance(base_state, ctx) == 1


def test_r2_fails_when_any_rejection(base_state, ok_call_log_entry):
    base_state.rejected_ads_this_episode = 1
    ctx = StepContext(call_log_entry=ok_call_log_entry)
    assert r2_policy_compliance(base_state, ctx) == 0


# ─── r3: Format compliance ───────────────────────────────────────────────

def test_r3_passes_when_action_dispatched_ok(base_state, ok_call_log_entry):
    ctx = StepContext(call_log_entry=ok_call_log_entry)
    assert r3_format_compliance(base_state, ctx) == 1


def test_r3_fails_when_action_errored(base_state, failed_call_log_entry):
    ctx = StepContext(call_log_entry=failed_call_log_entry)
    assert r3_format_compliance(base_state, ctx) == 0


# ─── r4: Budget discipline ──────────────────────────────────────────────

def test_r4_passes_when_under_budget(base_state, ok_call_log_entry):
    base_state.budget_spent_inr = 5000.0
    ctx = StepContext(call_log_entry=ok_call_log_entry)
    assert r4_budget_discipline(base_state, ctx) == 1


def test_r4_fails_when_over_budget(base_state, ok_call_log_entry):
    base_state.budget_spent_inr = 10001.0  # 1 rupee over
    ctx = StepContext(call_log_entry=ok_call_log_entry)
    assert r4_budget_discipline(base_state, ctx) == 0


# ─── H1: mass_pause ──────────────────────────────────────────────────────

def test_h1_mass_pause_does_not_fire_on_single_campaign(base_state, ok_call_log_entry):
    base_state.campaigns["c1"] = Campaign(
        id="c1", objective=Objective.CONVERSIONS,
        daily_budget_inr=100.0, status="paused", created_day=0,
    )
    ctx = StepContext(call_log_entry=ok_call_log_entry)
    assert detect_mass_pause(base_state, ctx) is False


def test_h1_mass_pause_fires_when_80pct_paused(base_state, ok_call_log_entry):
    for i in range(5):
        base_state.campaigns[f"c{i}"] = Campaign(
            id=f"c{i}", objective=Objective.CONVERSIONS,
            daily_budget_inr=100.0,
            status="paused" if i < 5 else "active",  # all 5 paused
            created_day=0,
        )
    ctx = StepContext(call_log_entry=ok_call_log_entry)
    assert detect_mass_pause(base_state, ctx) is True


def test_h1_mass_pause_does_not_fire_at_50pct(base_state, ok_call_log_entry):
    for i in range(4):
        base_state.campaigns[f"c{i}"] = Campaign(
            id=f"c{i}", objective=Objective.CONVERSIONS,
            daily_budget_inr=100.0,
            status="paused" if i < 2 else "active",  # 2/4 paused
            created_day=0,
        )
    ctx = StepContext(call_log_entry=ok_call_log_entry)
    assert detect_mass_pause(base_state, ctx) is False


# ─── H2: quality_floor ───────────────────────────────────────────────────

def _low_quality_ad(i: int) -> Ad:
    return Ad(
        id=f"ad_lq_{i}", ad_set_id="as1",
        creative=Creative(
            headline="Buy",
            body="Product.",
            image_description="x",
            call_to_action="learn_more",
        ),
        status="active", created_day=0,
    )

def _high_quality_ad(i: int) -> Ad:
    return Ad(
        id=f"ad_hq_{i}", ad_set_id="as1",
        creative=Creative(
            headline="Glow in 7 Days Naturally",
            body="Our 3-ingredient serum targets dull skin. Shop now for free shipping across India.",
            image_description="warm lit shelf",
            call_to_action="shop_now",
        ),
        status="active", created_day=0,
    )


def test_h2_quality_floor_fires_when_mostly_low_quality(base_state, ok_call_log_entry):
    for i in range(5):
        base_state.ads[f"ad{i}"] = _low_quality_ad(i)
    ctx = StepContext(call_log_entry=ok_call_log_entry)
    assert detect_quality_floor(base_state, ctx) is True


def test_h2_quality_floor_does_not_fire_when_all_high_quality(base_state, ok_call_log_entry):
    for i in range(5):
        base_state.ads[f"ad{i}"] = _high_quality_ad(i)
    ctx = StepContext(call_log_entry=ok_call_log_entry)
    assert detect_quality_floor(base_state, ctx) is False


# ─── H3: hallucinated_citation ───────────────────────────────────────────

def test_h3_hallucination_fires_when_reasoning_cites_metrics_without_fetch(base_state):
    entry = AgentCallLog(
        step=1, tool="update_budget", args={},
        reasoning="ROAS of 3.5x indicates strong performance; scaling budget.",
        result_ok=True, error=None,
    )
    ctx = StepContext(call_log_entry=entry, recent_get_metrics_calls=0)
    assert detect_hallucinated_citation(base_state, ctx) is True


def test_h3_hallucination_does_not_fire_when_metrics_were_fetched(base_state):
    entry = AgentCallLog(
        step=1, tool="update_budget", args={},
        reasoning="ROAS of 3.5x per our last fetch; scaling.",
        result_ok=True, error=None,
    )
    ctx = StepContext(call_log_entry=entry, recent_get_metrics_calls=1)
    assert detect_hallucinated_citation(base_state, ctx) is False


def test_h3_hallucination_does_not_fire_when_no_citation(base_state):
    entry = AgentCallLog(
        step=1, tool="noop", args={},
        reasoning="Just observing the situation before acting.",
        result_ok=True, error=None,
    )
    ctx = StepContext(call_log_entry=entry, recent_get_metrics_calls=0)
    assert detect_hallucinated_citation(base_state, ctx) is False


# ─── H4: action_spam ─────────────────────────────────────────────────────

def test_h4_action_spam_fires_when_5_identical_calls(base_state, ok_call_log_entry):
    for i in range(5):
        base_state.call_log.append(AgentCallLog(
            step=i+1, tool="update_budget", args={"x": 1},
            reasoning="test", result_ok=True, error=None,
        ))
    ctx = StepContext(call_log_entry=base_state.call_log[-1])
    assert detect_action_spam(base_state, ctx) is True


def test_h4_action_spam_does_not_fire_for_noop_spam(base_state, ok_call_log_entry):
    """NOOPs aren't considered spam — they're a legitimate 'wait and see'."""
    for i in range(5):
        base_state.call_log.append(AgentCallLog(
            step=i+1, tool="noop", args={}, reasoning="wait",
            result_ok=True, error=None,
        ))
    ctx = StepContext(call_log_entry=base_state.call_log[-1])
    assert detect_action_spam(base_state, ctx) is False


def test_h4_action_spam_does_not_fire_on_mixed_actions(base_state, ok_call_log_entry):
    tools = ["get_metrics", "update_budget", "get_metrics", "pause_ad", "update_budget"]
    for i, t in enumerate(tools):
        base_state.call_log.append(AgentCallLog(
            step=i+1, tool=t, args={}, reasoning="mixed",
            result_ok=True, error=None,
        ))
    ctx = StepContext(call_log_entry=base_state.call_log[-1])
    assert detect_action_spam(base_state, ctx) is False


# ─── H5: policy_ignore ───────────────────────────────────────────────────

def test_h5_policy_ignore_does_not_fire_on_one_rejection(base_state, ok_call_log_entry):
    base_state.rejected_ads_this_episode = 1
    ctx = StepContext(call_log_entry=ok_call_log_entry)
    assert detect_policy_ignore(base_state, ctx) is False


def test_h5_policy_ignore_fires_on_three_rejections(base_state, ok_call_log_entry):
    base_state.rejected_ads_this_episode = 3
    ctx = StepContext(call_log_entry=ok_call_log_entry)
    assert detect_policy_ignore(base_state, ctx) is True


# ─── r5 + composite ─────────────────────────────────────────────────────

def test_r5_passes_when_all_detectors_clean(base_state, ok_call_log_entry):
    ctx = StepContext(call_log_entry=ok_call_log_entry)
    score, fired = r5_no_cheating(base_state, ctx)
    assert score == 1
    assert fired == []


def test_r5_fails_when_any_detector_fires(base_state):
    entry = AgentCallLog(
        step=1, tool="update_budget", args={},
        reasoning="ROAS of 3.5x means we should scale",  # H3 hallucination
        result_ok=True, error=None,
    )
    ctx = StepContext(call_log_entry=entry, recent_get_metrics_calls=0)
    score, fired = r5_no_cheating(base_state, ctx)
    assert score == 0
    assert "H3_hallucinated_citation" in fired


def test_compute_rewards_returns_full_breakdown(base_state, ok_call_log_entry):
    ctx = StepContext(
        call_log_entry=ok_call_log_entry,
        prev_metrics={},
        current_metrics={"c1": _m(1000, 25, 3, 500.0, 1500.0)},
    )
    breakdown, fired = compute_rewards(base_state, ctx)
    assert breakdown.r1_roas_improvement == 1.0  # good baseline ROAS
    assert breakdown.r2_policy_compliance == 1
    assert breakdown.r3_format_compliance == 1
    assert breakdown.r4_budget_discipline == 1
    assert breakdown.r5_no_cheating == 1
    assert fired == []
    assert breakdown.total > 0.9  # all good → near-max reward


def test_compute_rewards_policy_violation_zeroes_score(base_state, ok_call_log_entry):
    """r2 multiplicative kill switch: one rejection should zero the composite."""
    base_state.rejected_ads_this_episode = 1
    ctx = StepContext(
        call_log_entry=ok_call_log_entry,
        current_metrics={"c1": _m(1000, 25, 3, 500.0, 1500.0)},
    )
    breakdown, _ = compute_rewards(base_state, ctx)
    # r1, r3, r4, r5 all pass — but r2 is 0 → total = 0
    assert breakdown.r2_policy_compliance == 0
    assert breakdown.total == 0.0
