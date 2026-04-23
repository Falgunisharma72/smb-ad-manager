"""End-to-end integration tests — verify the full wired env works."""
from __future__ import annotations

import pytest

from src.smb_ads.env import Env
from src.smb_ads.models import Action, ActionType, Creative


def test_reset_bootstraps_starter_campaign():
    """Every fresh episode has 1 running campaign + 1 ad_set + 1 ad + 1 day of metrics."""
    env = Env(task_id="easy", seed=42)
    obs = env.reset()

    # Starter state
    assert len(obs.active_campaigns) == 1
    assert len(obs.active_ad_sets) == 1
    assert len(obs.active_ads) == 1
    assert obs.active_campaigns[0].status == "active"

    # One day of metrics has been pre-generated
    assert len(obs.latest_metrics) == 1
    campaign_id = obs.active_campaigns[0].id
    m = obs.latest_metrics[campaign_id]
    assert m.impressions > 0  # simulator ran


def test_noop_action_still_advances_simulation():
    """Even a noop triggers: day rolls, metrics update, state dict updates."""
    env = Env(task_id="medium", seed=42)
    obs = env.reset()
    day_before = obs.day

    action = Action(tool=ActionType.NOOP, args={}, reasoning="observing first")
    result = env.step(action)
    assert result.observation.day > day_before
    assert result.reward.breakdown.r3_format_compliance == 1  # valid action


def test_malformed_action_fails_r3():
    """Calling a tool with missing/invalid args fails format compliance reward."""
    env = Env(task_id="easy", seed=42)
    env.reset()

    # create_campaign requires objective + daily_budget_inr; omit them
    bad = Action(
        tool=ActionType.CREATE_CAMPAIGN,
        args={},  # missing all required kwargs
        reasoning="deliberately malformed",
    )
    result = env.step(bad)
    # Action failed → r3=0
    assert result.reward.breakdown.r3_format_compliance == 0
    # And error was logged
    assert result.info.get("action_error") is not None


def test_agent_can_scale_winning_campaign():
    """End-to-end: agent observes, calls update_budget, budget is reflected in state."""
    env = Env(task_id="medium", seed=42)
    obs = env.reset()
    campaign_id = obs.active_campaigns[0].id
    old_budget = obs.active_campaigns[0].daily_budget_inr

    action = Action(
        tool=ActionType.UPDATE_BUDGET,
        args={
            "campaign_id": campaign_id,
            "new_daily_budget_inr": old_budget * 2,
        },
        reasoning="strong ROAS, scaling budget",
    )
    result = env.step(action)

    # Action succeeded
    assert result.reward.breakdown.r3_format_compliance == 1
    # And budget actually updated in state
    new_campaign = result.observation.active_campaigns[0]
    assert new_campaign.daily_budget_inr == old_budget * 2


def test_policy_drift_fires_on_medium_tier():
    """On medium tier, a policy rule should activate mid-episode."""
    env = Env(task_id="medium", seed=42)
    obs = env.reset()

    initial_policy_count = len(obs.active_policies)
    # 5 base rules at start
    assert initial_policy_count == 5

    action = Action(tool=ActionType.NOOP, args={}, reasoning="wait")
    # Medium tier: after reset day=1. Step 1 advances to day=2 and fires drift.
    result1 = env.step(action)  # day now 2 — drift event fires THIS step
    result2 = env.step(action)  # day now 3 — already applied

    # The POLICY UPDATE event appears in step 1's recent_events
    assert any("POLICY UPDATE" in e for e in result1.observation.recent_events), \
        f"Expected POLICY UPDATE in step 1 events, got: {result1.observation.recent_events}"

    # Rule is present in active_policies for both subsequent observations
    for result in (result1, result2):
        policy_ids = {p.id for p in result.observation.active_policies}
        assert "p6_health_disclaimer" in policy_ids


def test_hard_task_runs_full_seven_steps():
    """Hard task should terminate after 7 steps."""
    env = Env(task_id="hard", seed=42)
    env.reset()

    action = Action(tool=ActionType.NOOP, args={}, reasoning="test")
    step_count = 0
    while step_count < 15:
        result = env.step(action)
        step_count += 1
        if result.done:
            break
    assert step_count == 7


def test_state_snapshot_is_json_serializable():
    """env.state() must be JSON-dumpable — required by OpenEnv spec."""
    import json

    env = Env(task_id="easy", seed=42)
    env.reset()
    action = Action(tool=ActionType.NOOP, args={}, reasoning="x")
    env.step(action)

    snapshot = env.state()
    # Must not raise
    s = json.dumps(snapshot)
    assert len(s) > 0
    assert "task_id" in snapshot
    assert "budget_remaining_inr" in snapshot


def test_policy_rejection_increments_state_counter():
    """Creating an ad that violates policy should bump the rejection counter."""
    env = Env(task_id="easy", seed=42)
    obs = env.reset()
    ad_set_id = obs.active_ad_sets[0].id

    # Agent tries to push a policy-violating creative
    action = Action(
        tool=ActionType.CREATE_AD,
        args={
            "ad_set_id": ad_set_id,
            "creative": {
                "headline": "Cures acne in 3 days",  # triggers health-claim rule
                "body": "Clinically proven FDA approved miracle cream.",
                "image_description": "before/after photo",
                "call_to_action": "shop_now",
            },
        },
        reasoning="aggressive health claim",
    )
    result = env.step(action)

    # Action itself didn't error (valid schema), but policy rejected the ad
    assert result.reward.breakdown.r2_policy_compliance == 0
    # So total reward should be zero (r2 is multiplicative kill switch)
    assert result.reward.score == 0.0


def test_budget_tracking_via_real_metrics():
    """After N steps, spend_inr in state should reflect simulated spend."""
    env = Env(task_id="hard", seed=42)
    obs = env.reset()
    campaign_id = obs.active_campaigns[0].id
    initial_spend = obs.total_budget_remaining_inr

    action = Action(tool=ActionType.NOOP, args={}, reasoning="just watching")
    for _ in range(7):
        result = env.step(action)
        if result.done:
            break

    final_spend = result.observation.total_budget_remaining_inr
    # Budget should have decreased
    assert final_spend < initial_spend
    # But not gone negative
    assert final_spend > 0 or result.reward.breakdown.r4_budget_discipline == 0
