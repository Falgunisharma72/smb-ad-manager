"""Pydantic schemas for the SMB Ad Manager environment.

These are the contract between the agent and the environment. Strict typing
(no extras allowed) is intentional — schema violations are a reward signal
(format_compliance in rewards/).
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Pydantic config shared by all schemas: forbid extra fields so we catch
# agents inventing keys. This is part of our anti-hack design.
STRICT = ConfigDict(extra="forbid", frozen=False)


# ─── Industry / SMB profile ──────────────────────────────────────────────

class Industry(str, Enum):
    """The 3 industries we calibrate user-response data for."""
    SKINCARE = "skincare"
    FOOD_DELIVERY = "food_delivery"
    FITNESS_APPS = "fitness_apps"


class SMBProfile(BaseModel):
    """A small business the agent is helping."""
    model_config = STRICT

    name: str
    industry: Industry
    location: str  # e.g., "Mumbai, India"
    monthly_budget_inr: float = Field(..., gt=0)
    goal: Literal["online_sales", "brand_awareness", "lead_gen", "app_installs"]
    description: str  # free-text pitch of the business


# ─── Campaign / AdSet / Ad / Creative ────────────────────────────────────

class Objective(str, Enum):
    CONVERSIONS = "conversions"
    TRAFFIC = "traffic"
    REACH = "reach"
    APP_INSTALLS = "app_installs"


class Campaign(BaseModel):
    model_config = STRICT
    id: str
    objective: Objective
    daily_budget_inr: float = Field(..., gt=0)
    status: Literal["active", "paused", "draft"]
    created_day: int  # day of simulation when created


class AdSet(BaseModel):
    model_config = STRICT
    id: str
    campaign_id: str
    audience_description: str  # e.g., "women 25-45 interested in skincare, India"
    status: Literal["active", "paused", "draft"]


class Creative(BaseModel):
    """The actual ad content. Simplified: text + image_description."""
    model_config = STRICT
    headline: str = Field(..., max_length=120)
    body: str = Field(..., max_length=500)
    image_description: str  # in lieu of real image upload
    call_to_action: Literal["shop_now", "learn_more", "sign_up", "install"]


class Ad(BaseModel):
    model_config = STRICT
    id: str
    ad_set_id: str
    creative: Creative
    status: Literal["active", "paused", "rejected"]
    rejection_reason: Optional[str] = None  # if policy rejected
    created_day: int


# ─── Metrics ─────────────────────────────────────────────────────────────

class Metrics(BaseModel):
    """Performance snapshot, returned by get_metrics tool."""
    model_config = STRICT

    impressions: int = Field(..., ge=0)
    clicks: int = Field(..., ge=0)
    conversions: int = Field(..., ge=0)
    spend_inr: float = Field(..., ge=0.0)
    revenue_inr: float = Field(..., ge=0.0)  # attributed revenue from conversions

    @property
    def ctr(self) -> float:
        return self.clicks / self.impressions if self.impressions else 0.0

    @property
    def conversion_rate(self) -> float:
        return self.conversions / self.clicks if self.clicks else 0.0

    @property
    def roas(self) -> float:
        return self.revenue_inr / self.spend_inr if self.spend_inr else 0.0


# ─── Policy ──────────────────────────────────────────────────────────────

class PolicyRule(BaseModel):
    """One policy constraint the enforcer checks against."""
    model_config = STRICT
    id: str
    name: str
    description: str
    active_since_day: int  # for schema drift — rules activate over time


# ─── Actions (the agent's vocabulary) ────────────────────────────────────

class ActionType(str, Enum):
    CREATE_CAMPAIGN = "create_campaign"
    CREATE_AD_SET = "create_ad_set"
    CREATE_AD = "create_ad"
    GET_METRICS = "get_metrics"
    UPDATE_BUDGET = "update_budget"
    PAUSE_AD = "pause_ad"
    GET_POLICY_UPDATES = "get_policy_updates"
    REWRITE_CREATIVE = "rewrite_creative"
    NOOP = "noop"  # explicit wait-and-watch


class Action(BaseModel):
    """Every agent turn produces one of these. Strict — violations tank r3."""
    model_config = STRICT

    tool: ActionType
    args: dict = Field(default_factory=dict)  # tool-specific; validated per-tool elsewhere
    reasoning: str = Field(..., min_length=1, max_length=2000)


# ─── Observation (what the agent sees each step) ─────────────────────────

class Observation(BaseModel):
    """State snapshot the env returns after each step."""
    model_config = STRICT

    step: int = Field(..., ge=0)
    day: int = Field(..., ge=0)  # simulation day (for long-horizon reasoning)
    task_id: Literal["easy", "medium", "hard"]

    # Business context (shown once at reset, can repeat)
    smb_profile: SMBProfile
    total_budget_remaining_inr: float
    days_elapsed: int
    days_total: int

    # Current state of active resources
    active_campaigns: list[Campaign]
    active_ad_sets: list[AdSet]
    active_ads: list[Ad]

    # Recent events (last step's outputs, policy updates, etc.)
    recent_events: list[str] = Field(default_factory=list)

    # Latest metrics per campaign, if any get_metrics call happened recently
    latest_metrics: dict[str, Metrics] = Field(default_factory=dict)  # keyed by campaign_id

    # Policy context (what rules are currently in effect)
    active_policies: list[PolicyRule] = Field(default_factory=list)

    # Available tools list (helps small models stay formatted)
    available_tools: list[ActionType] = Field(default_factory=lambda: list(ActionType))


# ─── Reward (what comes back from step) ──────────────────────────────────

class RewardBreakdown(BaseModel):
    """The 5 independent reward components — logged separately on wandb."""
    model_config = STRICT

    r1_roas_improvement: float = Field(..., ge=0.0, le=1.0)
    r2_policy_compliance: int = Field(..., ge=0, le=1)
    r3_format_compliance: int = Field(..., ge=0, le=1)
    r4_budget_discipline: int = Field(..., ge=0, le=1)
    r5_no_cheating: int = Field(..., ge=0, le=1)

    @property
    def total(self) -> float:
        """Training uses this sum; individual columns are logged separately."""
        # r2 is multiplicative kill switch — one policy violation zeroes everything
        if self.r2_policy_compliance == 0:
            return 0.0
        return (
            self.r1_roas_improvement
            + self.r3_format_compliance
            + self.r4_budget_discipline
            + self.r5_no_cheating
        ) / 4.0


class Reward(BaseModel):
    """Returned from Env.step()."""
    model_config = STRICT

    score: float = Field(..., ge=0.0, le=1.0)
    done: bool = False
    breakdown: RewardBreakdown
    info: dict = Field(default_factory=dict)  # any extra debug info


# ─── Step result tuple ───────────────────────────────────────────────────

class StepResult(BaseModel):
    """Packaged return from env.step(). Matches OpenEnv convention."""
    model_config = STRICT

    observation: Observation
    reward: Reward
    done: bool
    info: dict = Field(default_factory=dict)
