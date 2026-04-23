"""OpenEnv-compliant Env class.

At this stage (Hour 0-4 of the build), this is a SKELETON — step/reset return
dummy values so we can verify the HF Space deploys and the API wiring works.
Real simulation logic gets filled in Hours 4-10.

Usage contract (per OpenEnv spec):
    env = Env(task_id="easy", seed=42)
    obs = env.reset()
    obs, reward, done, info = env.step(action)
    state_dict = env.state()  # serializable snapshot

Structured stdout logs: [START], [STEP], [END] (mandatory for hackathon grading).
"""
from __future__ import annotations

import json
from typing import Literal, Optional

from .models import (
    Action,
    Industry,
    Metrics,
    Observation,
    Reward,
    RewardBreakdown,
    SMBProfile,
    StepResult,
)


class Env:
    """Skeleton OpenEnv environment. Real logic lands in Hours 4-10."""

    def __init__(
        self,
        task_id: Literal["easy", "medium", "hard"] = "easy",
        seed: Optional[int] = None,
    ):
        self._task_id = task_id
        self._seed = seed
        self._step_count = 0
        self._day = 0
        self._done = False

        # Placeholder SMB profile — replaced by real generator in scenarios.py
        self._smb = SMBProfile(
            name="Priya's Handmade Candles",
            industry=Industry.SKINCARE,  # TODO: swap to proper industry when generator lands
            location="Mumbai, India",
            monthly_budget_inr=10000.0,
            goal="online_sales",
            description="Handmade soy candles, natural scents, targeting urban millennials.",
        )
        self._budget_remaining = self._smb.monthly_budget_inr
        self._days_total = {"easy": 1, "medium": 3, "hard": 7}[task_id]

    # ─── OpenEnv-required methods ────────────────────────────────────────

    def reset(self) -> Observation:
        """Start a fresh episode. Resets state + prints [START] log."""
        self._step_count = 0
        self._day = 0
        self._done = False
        self._budget_remaining = self._smb.monthly_budget_inr

        print(
            f"[START] task={self._task_id} seed={self._seed} "
            f"smb={self._smb.name!r} budget={self._budget_remaining:.2f} "
            f"horizon_days={self._days_total}",
            flush=True,
        )

        return self._make_observation()

    def step(self, action: Action) -> StepResult:
        """Apply one action, return next observation + reward."""
        self._step_count += 1

        # Placeholder reward — real 5-component reward lands in Hour 8-10
        breakdown = RewardBreakdown(
            r1_roas_improvement=0.5,  # TODO: real ROAS calc
            r2_policy_compliance=1,   # TODO: real policy check
            r3_format_compliance=1,   # action was parsed OK by pydantic
            r4_budget_discipline=1,   # TODO: real budget check
            r5_no_cheating=1,         # TODO: real anti-hack detectors
        )
        reward = Reward(
            score=breakdown.total,
            done=False,
            breakdown=breakdown,
            info={"placeholder": True},
        )

        # Advance simulated day
        if self._step_count >= self._days_total:
            self._done = True
            self._day = self._days_total
        else:
            self._day = self._step_count

        print(
            f"[STEP] action={action.tool.value} score={reward.score:.3f} "
            f"r1={breakdown.r1_roas_improvement:.2f} r2={breakdown.r2_policy_compliance} "
            f"r3={breakdown.r3_format_compliance} r4={breakdown.r4_budget_discipline} "
            f"r5={breakdown.r5_no_cheating} done={self._done}",
            flush=True,
        )

        if self._done:
            print(
                f"[END] task={self._task_id} final_score={reward.score:.3f} "
                f"steps={self._step_count}",
                flush=True,
            )

        return StepResult(
            observation=self._make_observation(),
            reward=reward,
            done=self._done,
            info=reward.info,
        )

    def state(self) -> dict:
        """Serializable snapshot for checkpointing/grading."""
        return {
            "task_id": self._task_id,
            "seed": self._seed,
            "step": self._step_count,
            "day": self._day,
            "done": self._done,
            "budget_remaining": self._budget_remaining,
            "smb": self._smb.model_dump(),
        }

    # ─── Helpers ─────────────────────────────────────────────────────────

    def _make_observation(self) -> Observation:
        """Construct an Observation from current state. Skeleton version."""
        return Observation(
            step=self._step_count,
            day=self._day,
            task_id=self._task_id,
            smb_profile=self._smb,
            total_budget_remaining_inr=self._budget_remaining,
            days_elapsed=self._day,
            days_total=self._days_total,
            active_campaigns=[],
            active_ad_sets=[],
            active_ads=[],
            recent_events=[f"Simulation day {self._day}"],
            latest_metrics={},
            active_policies=[],
        )
