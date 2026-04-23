"""OpenEnv-compliant Env class — the real simulation.

This is where everything connects:
  - AccountState holds the in-memory world.
  - The agent's Action is dispatched through the Marketing API.
  - After each step, a simulated "day" passes — user_model generates metrics,
    policy drift may activate new rules.
  - RewardBreakdown is placeholder here; real 5-reward logic lands next commit.

Structured stdout logs: [START], [STEP], [END] (mandatory for hackathon grading).

Reset behavior: each episode starts with one campaign already running for 1 day,
so the agent is always in a "campaign reaction" position — must diagnose the
current state and act. This matches the narrow task design in the locked plan.
"""
from __future__ import annotations

import json
from typing import Literal, Optional

from . import policy, policy_drift, scenarios, user_model
from .marketing_api import ToolError, dispatch
from .models import (
    Action,
    ActionType,
    Campaign,
    Creative,
    Industry,
    Metrics,
    Objective,
    Observation,
    Reward,
    RewardBreakdown,
    StepResult,
)
from .state import AccountState, AgentCallLog


# ─── Industry-specific starter creatives ────────────────────────────────
# Each episode begins with one of these as the pre-existing ad. The agent
# then decides: scale / pause / rewrite / ignore.

STARTER_CREATIVES: dict[Industry, Creative] = {
    Industry.SKINCARE: Creative(
        headline="Glow Naturally This Season",
        body="Our handcrafted skincare blends. Shop now and feel the difference.",
        image_description="amber glass bottle on warm wooden surface",
        call_to_action="shop_now",
    ),
    Industry.FOOD_DELIVERY: Creative(
        headline="Home-Style Meals Delivered",
        body="Fresh, home-cooked dinners delivered to your door. Order now.",
        image_description="thali tray with steam rising",
        call_to_action="shop_now",
    ),
    Industry.FITNESS_APPS: Creative(
        headline="Fitness That Fits You",
        body="Personalized workout plans for busy Indian professionals. Install now.",
        image_description="person jogging at sunrise",
        call_to_action="install",
    ),
}


# How many simulation days each task tier runs
_DAYS_BY_TIER = {"easy": 1, "medium": 3, "hard": 7}


class Env:
    """OpenEnv-compliant SMB Ad Manager environment.

    Usage:
        env = Env(task_id="easy", seed=42)
        obs = env.reset()
        while not done:
            action = agent_policy(obs)
            result = env.step(action)
            obs, done = result.observation, result.done
    """

    def __init__(
        self,
        task_id: Literal["easy", "medium", "hard"] = "easy",
        seed: Optional[int] = None,
    ):
        if task_id not in _DAYS_BY_TIER:
            raise ValueError(f"Invalid task_id: {task_id}")
        self._task_id = task_id
        self._seed = seed
        self._days_total = _DAYS_BY_TIER[task_id]

        # State populated on reset()
        self._state: Optional[AccountState] = None
        self._step_count = 0
        self._day = 0
        self._done = False
        self._pending_events: list[str] = []

    # ─── OpenEnv-required methods ────────────────────────────────────────

    def reset(self) -> Observation:
        """Start a new episode: SMB + 1 running campaign + 1 day of metrics."""
        # Pick an SMB profile
        smb = scenarios.get_smb_profile(seed=self._seed)

        # Fresh state
        self._state = AccountState(
            smb=smb,
            total_budget_inr=smb.monthly_budget_inr,
            active_policies={p.id: p for p in policy_drift.initial_policies(self._task_id)},
        )
        self._step_count = 0
        self._day = 0
        self._done = False
        self._pending_events = []

        # Pre-create a starter campaign so the agent is always in "react mode"
        self._bootstrap_starter_campaign()

        # Advance to day 1 with one day of metrics already collected
        self._day = 1
        self._run_daily_simulation(seed_offset=0)

        print(
            f"[START] task={self._task_id} seed={self._seed} smb={smb.name!r} "
            f"industry={smb.industry.value} budget={smb.monthly_budget_inr:.0f} "
            f"horizon_days={self._days_total}",
            flush=True,
        )

        return self._make_observation(
            recent_events=[
                f"Starter campaign running for 1 day; review performance and decide next action."
            ],
        )

    def step(self, action: Action) -> StepResult:
        """Apply one action, simulate the next day, return new observation + reward."""
        assert self._state is not None, "Must call reset() before step()"
        self._step_count += 1

        # ─── Dispatch the action ────────────────────────────────────────
        call_ok = True
        error: Optional[str] = None
        result_dict: dict = {}
        checker = policy.make_policy_checker(
            self._state.active_policies.values(),
            current_day=self._day,
        )
        try:
            result_dict = dispatch(
                tool_name=action.tool.value,
                state=self._state,
                day=self._day,
                args=action.args,
                policy_checker=checker,
            )
        except ToolError as e:
            call_ok = False
            error = str(e)
            self._state.malformed_action_count += 1
        except TypeError as e:
            # Agent passed wrong kwargs
            call_ok = False
            error = f"Invalid args: {e}"
            self._state.malformed_action_count += 1

        # Log the call (drives anti-hack detection)
        self._state.call_log.append(
            AgentCallLog(
                step=self._step_count,
                tool=action.tool.value,
                args=action.args,
                reasoning=action.reasoning,
                result_ok=call_ok,
                error=error,
            )
        )

        # ─── Advance one day + apply policy drift ───────────────────────
        self._day += 1

        policies_dict, drift_events = policy_drift.apply_drift(
            dict(self._state.active_policies),
            self._task_id,
            current_day=self._day,
        )
        self._state.active_policies = policies_dict

        # ─── Simulate a day of performance ──────────────────────────────
        self._run_daily_simulation(seed_offset=self._step_count)

        # ─── Determine if episode is done ──────────────────────────────
        if self._step_count >= self._days_total:
            self._done = True

        # ─── Build reward (PLACEHOLDER — real 5-reward logic lands next) ─
        breakdown = RewardBreakdown(
            r1_roas_improvement=min(1.0, self._latest_campaign_roas() / 2.0),
            r2_policy_compliance=1 if self._state.rejected_ads_this_episode == 0 else 0,
            r3_format_compliance=1 if call_ok else 0,
            r4_budget_discipline=1 if self._state.budget_spent_inr <= self._state.total_budget_inr else 0,
            r5_no_cheating=1,  # proper anti-hack detectors land next commit
        )
        reward = Reward(
            score=breakdown.total,
            done=self._done,
            breakdown=breakdown,
            info={
                "action_ok": call_ok,
                "action_error": error,
                "tool_result": result_dict,
            },
        )

        # ─── Build recent_events for the next observation ───────────────
        events = list(drift_events)
        if error:
            events.append(f"Action '{action.tool.value}' FAILED: {error}")
        else:
            events.append(f"Action '{action.tool.value}' executed OK")

        # ─── Log ────────────────────────────────────────────────────────
        print(
            f"[STEP {self._step_count}] day={self._day} action={action.tool.value} "
            f"ok={call_ok} score={reward.score:.3f} "
            f"r1={breakdown.r1_roas_improvement:.2f} r2={breakdown.r2_policy_compliance} "
            f"r3={breakdown.r3_format_compliance} r4={breakdown.r4_budget_discipline} "
            f"r5={breakdown.r5_no_cheating} done={self._done}",
            flush=True,
        )
        if self._done:
            print(
                f"[END] task={self._task_id} final_score={reward.score:.3f} "
                f"steps={self._step_count} total_spend={self._state.budget_spent_inr:.2f}",
                flush=True,
            )

        return StepResult(
            observation=self._make_observation(recent_events=events),
            reward=reward,
            done=self._done,
            info=reward.info,
        )

    def state(self) -> dict:
        """Serializable state snapshot (for grading / debugging)."""
        if self._state is None:
            return {"status": "not_initialized"}
        base = self._state.to_dict()
        base.update({
            "task_id": self._task_id,
            "seed": self._seed,
            "step": self._step_count,
            "day": self._day,
            "done": self._done,
        })
        return base

    # ─── Internals ───────────────────────────────────────────────────────

    def _bootstrap_starter_campaign(self) -> None:
        """Create the pre-existing campaign / ad_set / ad for this episode."""
        assert self._state is not None
        smb = self._state.smb

        # Pick objective based on SMB goal
        objective_map = {
            "online_sales": Objective.CONVERSIONS,
            "brand_awareness": Objective.REACH,
            "lead_gen": Objective.TRAFFIC,
            "app_installs": Objective.APP_INSTALLS,
        }
        objective = objective_map.get(smb.goal, Objective.CONVERSIONS)

        # Daily budget ~ 1/30 of monthly
        daily = smb.monthly_budget_inr / 30.0

        cid = self._state.next_id("campaign")
        self._state.campaigns[cid] = Campaign(
            id=cid,
            objective=objective,
            daily_budget_inr=round(daily, 2),
            status="active",
            created_day=0,
        )

        asid = self._state.next_id("ad_set")
        from .models import AdSet
        self._state.ad_sets[asid] = AdSet(
            id=asid,
            campaign_id=cid,
            audience_description=_default_audience_for(smb.industry),
            status="active",
        )

        ad_id = self._state.next_id("ad")
        from .models import Ad
        self._state.ads[ad_id] = Ad(
            id=ad_id,
            ad_set_id=asid,
            creative=STARTER_CREATIVES[smb.industry],
            status="active",
            rejection_reason=None,
            created_day=0,
        )

    def _run_daily_simulation(self, *, seed_offset: int = 0) -> None:
        """Simulate one day of performance across all active campaigns."""
        assert self._state is not None
        for campaign in list(self._state.campaigns.values()):
            if campaign.status != "active":
                continue

            # Collect this campaign's active ads + audience info
            campaign_ads = []
            audience_descriptions: dict[str, str] = {}
            for ad in self._state.ads.values():
                ad_set = self._state.ad_sets.get(ad.ad_set_id)
                if not ad_set:
                    continue
                if ad_set.campaign_id != campaign.id:
                    continue
                if ad.status != "active":
                    continue
                campaign_ads.append(ad)
                audience_descriptions[ad_set.id] = ad_set.audience_description

            if not campaign_ads:
                continue

            prior = {
                ad.id: self._state.most_recent_metrics(campaign.id) or Metrics(
                    impressions=0, clicks=0, conversions=0,
                    spend_inr=0.0, revenue_inr=0.0,
                )
                for ad in campaign_ads
            }

            seed = None
            if self._seed is not None:
                seed = self._seed + seed_offset + sum(ord(c) for c in campaign.id)

            daily_metrics = user_model.aggregate_campaign_metrics(
                ads=campaign_ads,
                audience_descriptions=audience_descriptions,
                daily_budget_inr=campaign.daily_budget_inr,
                industry=self._state.smb.industry,
                prior_metrics=prior,
                seed=seed,
            )

            self._state.metrics_history[(campaign.id, self._day)] = daily_metrics
            self._state.budget_spent_inr += daily_metrics.spend_inr

    def _latest_campaign_roas(self) -> float:
        """Return average ROAS across all campaigns' most recent metrics."""
        if self._state is None or not self._state.campaigns:
            return 0.0
        roas_values = []
        for c in self._state.campaigns.values():
            m = self._state.most_recent_metrics(c.id)
            if m and m.spend_inr > 0:
                roas_values.append(m.roas)
        return sum(roas_values) / len(roas_values) if roas_values else 0.0

    def _make_observation(self, *, recent_events: list[str]) -> Observation:
        """Build an Observation object from current state."""
        assert self._state is not None

        # Build latest_metrics dict (keyed by campaign_id)
        latest_metrics: dict[str, Metrics] = {}
        for c in self._state.campaigns.values():
            m = self._state.most_recent_metrics(c.id)
            if m is not None:
                latest_metrics[c.id] = m

        return Observation(
            step=self._step_count,
            day=self._day,
            task_id=self._task_id,
            smb_profile=self._state.smb,
            total_budget_remaining_inr=round(self._state.budget_remaining_inr, 2),
            days_elapsed=self._day,
            days_total=self._days_total,
            active_campaigns=list(self._state.campaigns.values()),
            active_ad_sets=list(self._state.ad_sets.values()),
            active_ads=list(self._state.ads.values()),
            recent_events=recent_events,
            latest_metrics=latest_metrics,
            active_policies=list(self._state.active_policies.values()),
        )


# ─── Helpers ────────────────────────────────────────────────────────────

def _default_audience_for(industry: Industry) -> str:
    """Default audience description per industry (placeholder for real targeting)."""
    return {
        Industry.SKINCARE: "Women 25-45 in Tier-1 Indian cities, interested in natural beauty products",
        Industry.FOOD_DELIVERY: "Urban professionals 22-40 ordering dinner 3+ times/week",
        Industry.FITNESS_APPS: "Active users 20-35 interested in fitness, living in urban India",
    }[industry]
