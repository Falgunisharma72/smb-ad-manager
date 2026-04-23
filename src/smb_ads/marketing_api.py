"""Mock Meta Marketing API — 8 tool endpoints the agent can call.

Each function mutates AccountState and returns a result dict. Errors raise
ToolError so the env can log a malformed-call (feeds r3 format_compliance).

Endpoints (per the locked plan):
    1. create_campaign(objective, daily_budget_inr) -> campaign_id
    2. create_ad_set(campaign_id, audience, targeting) -> ad_set_id
    3. create_ad(ad_set_id, creative) -> ad_id or rejection
    4. get_metrics(campaign_id) -> Metrics
    5. update_budget(campaign_id, new_daily_budget_inr)
    6. pause_ad(ad_id) or pause_campaign(campaign_id)
    7. get_policy_updates() -> active policies
    8. rewrite_creative(ad_id, new_creative) -> re-submits for review

The agent does NOT see this file — only the HTTP /step endpoint, which calls
into these via the Env class.
"""
from __future__ import annotations

from typing import Any

from .models import (
    Ad,
    AdSet,
    Campaign,
    Creative,
    Metrics,
    Objective,
    PolicyRule,
)
from .state import AccountState


class ToolError(Exception):
    """Raised when an agent action has invalid arguments. Non-fatal — env records it."""


# ─── Campaign tools ──────────────────────────────────────────────────────

def create_campaign(
    state: AccountState,
    day: int,
    *,
    objective: str,
    daily_budget_inr: float,
) -> dict[str, Any]:
    """Create a new campaign. Returns {campaign_id}."""
    try:
        obj = Objective(objective)
    except ValueError as e:
        raise ToolError(f"Invalid objective {objective!r}. Valid: {[o.value for o in Objective]}") from e

    if daily_budget_inr <= 0:
        raise ToolError(f"daily_budget_inr must be > 0, got {daily_budget_inr}")
    if daily_budget_inr > state.budget_remaining_inr:
        raise ToolError(
            f"daily_budget_inr {daily_budget_inr} exceeds remaining budget {state.budget_remaining_inr:.2f}"
        )

    cid = state.next_id("campaign")
    state.campaigns[cid] = Campaign(
        id=cid,
        objective=obj,
        daily_budget_inr=daily_budget_inr,
        status="active",
        created_day=day,
    )
    return {"campaign_id": cid, "status": "active"}


def create_ad_set(
    state: AccountState,
    day: int,
    *,
    campaign_id: str,
    audience_description: str,
) -> dict[str, Any]:
    """Create an ad set under a campaign. Returns {ad_set_id}."""
    if campaign_id not in state.campaigns:
        raise ToolError(f"Unknown campaign_id {campaign_id!r}")
    if len(audience_description.strip()) < 5:
        raise ToolError("audience_description too short")

    asid = state.next_id("ad_set")
    state.ad_sets[asid] = AdSet(
        id=asid,
        campaign_id=campaign_id,
        audience_description=audience_description,
        status="active",
    )
    return {"ad_set_id": asid, "status": "active"}


def create_ad(
    state: AccountState,
    day: int,
    *,
    ad_set_id: str,
    creative: dict,
    policy_checker=None,  # callable(creative) -> (ok: bool, reason: str)
) -> dict[str, Any]:
    """Create an ad with creative content. Runs policy check. Returns {ad_id, status, rejection_reason?}."""
    if ad_set_id not in state.ad_sets:
        raise ToolError(f"Unknown ad_set_id {ad_set_id!r}")

    # Validate creative schema via pydantic — this is where format bugs surface
    try:
        creative_obj = Creative(**creative)
    except Exception as e:
        raise ToolError(f"Invalid creative: {e}") from e

    # Policy check (if a checker was injected)
    rejection_reason = None
    status = "active"
    if policy_checker is not None:
        ok, reason = policy_checker(creative_obj, state.active_policies.values())
        if not ok:
            status = "rejected"
            rejection_reason = reason
            state.rejected_ads_this_episode += 1

    ad_id = state.next_id("ad")
    state.ads[ad_id] = Ad(
        id=ad_id,
        ad_set_id=ad_set_id,
        creative=creative_obj,
        status=status,
        rejection_reason=rejection_reason,
        created_day=day,
    )
    return {
        "ad_id": ad_id,
        "status": status,
        "rejection_reason": rejection_reason,
    }


# ─── Metrics / monitoring ────────────────────────────────────────────────

def get_metrics(
    state: AccountState,
    day: int,
    *,
    campaign_id: str,
) -> dict[str, Any]:
    """Fetch latest metrics for a campaign. Returns a Metrics dict."""
    if campaign_id not in state.campaigns:
        raise ToolError(f"Unknown campaign_id {campaign_id!r}")

    m = state.most_recent_metrics(campaign_id)
    if m is None:
        # No metrics recorded yet — return zeros (campaign hasn't run a cycle)
        m = Metrics(
            impressions=0, clicks=0, conversions=0,
            spend_inr=0.0, revenue_inr=0.0,
        )
    return m.model_dump() | {
        "campaign_id": campaign_id,
        "ctr": m.ctr,
        "conversion_rate": m.conversion_rate,
        "roas": m.roas,
    }


# ─── Budget / lifecycle ──────────────────────────────────────────────────

def update_budget(
    state: AccountState,
    day: int,
    *,
    campaign_id: str,
    new_daily_budget_inr: float,
) -> dict[str, Any]:
    """Adjust daily budget for a campaign."""
    if campaign_id not in state.campaigns:
        raise ToolError(f"Unknown campaign_id {campaign_id!r}")
    if new_daily_budget_inr <= 0:
        raise ToolError(f"new_daily_budget_inr must be > 0, got {new_daily_budget_inr}")
    # Note: we do NOT block over-budget here — that's enforced by r4_budget_discipline reward.
    # This way the agent can make the mistake and the reward catches it.
    c = state.campaigns[campaign_id]
    old = c.daily_budget_inr
    c.daily_budget_inr = new_daily_budget_inr
    return {"campaign_id": campaign_id, "old_budget": old, "new_budget": new_daily_budget_inr}


def pause_ad(
    state: AccountState,
    day: int,
    *,
    ad_id: str | None = None,
    campaign_id: str | None = None,
) -> dict[str, Any]:
    """Pause an ad OR an entire campaign. Exactly one target required."""
    if bool(ad_id) == bool(campaign_id):
        raise ToolError("Provide exactly one of ad_id or campaign_id")

    if ad_id is not None:
        if ad_id not in state.ads:
            raise ToolError(f"Unknown ad_id {ad_id!r}")
        state.ads[ad_id].status = "paused"
        return {"paused": "ad", "id": ad_id}

    if campaign_id is not None:
        if campaign_id not in state.campaigns:
            raise ToolError(f"Unknown campaign_id {campaign_id!r}")
        state.campaigns[campaign_id].status = "paused"
        # Also pause descendant ads (matches real Meta behavior)
        for ad in state.ads.values():
            if state.ad_sets.get(ad.ad_set_id) and state.ad_sets[ad.ad_set_id].campaign_id == campaign_id:
                if ad.status == "active":
                    ad.status = "paused"
        return {"paused": "campaign", "id": campaign_id}

    raise ToolError("unreachable")  # for the linter


# ─── Policy ──────────────────────────────────────────────────────────────

def get_policy_updates(
    state: AccountState,
    day: int,
) -> dict[str, Any]:
    """Return list of currently active policy rules."""
    return {
        "active_policies": [p.model_dump() for p in state.active_policies.values()],
        "count": len(state.active_policies),
    }


def rewrite_creative(
    state: AccountState,
    day: int,
    *,
    ad_id: str,
    new_creative: dict,
    policy_checker=None,
) -> dict[str, Any]:
    """Rewrite a rejected ad and resubmit for review."""
    if ad_id not in state.ads:
        raise ToolError(f"Unknown ad_id {ad_id!r}")

    try:
        new_creative_obj = Creative(**new_creative)
    except Exception as e:
        raise ToolError(f"Invalid creative: {e}") from e

    ad = state.ads[ad_id]
    # Re-run policy check
    status = "active"
    rejection_reason = None
    if policy_checker is not None:
        ok, reason = policy_checker(new_creative_obj, state.active_policies.values())
        if not ok:
            status = "rejected"
            rejection_reason = reason
            state.rejected_ads_this_episode += 1

    # Mutate the ad in place — keep the same ID (matches Meta's revision flow)
    ad.creative = new_creative_obj
    ad.status = status
    ad.rejection_reason = rejection_reason

    return {
        "ad_id": ad_id,
        "status": status,
        "rejection_reason": rejection_reason,
    }


# ─── Dispatcher — called by env.step() ───────────────────────────────────

TOOL_DISPATCH = {
    "create_campaign": create_campaign,
    "create_ad_set": create_ad_set,
    "create_ad": create_ad,
    "get_metrics": get_metrics,
    "update_budget": update_budget,
    "pause_ad": pause_ad,
    "get_policy_updates": get_policy_updates,
    "rewrite_creative": rewrite_creative,
}


def dispatch(
    tool_name: str,
    state: AccountState,
    day: int,
    args: dict,
    policy_checker=None,
) -> dict[str, Any]:
    """Execute a tool by name. Raises ToolError on bad args."""
    if tool_name == "noop":
        return {"status": "noop"}
    if tool_name not in TOOL_DISPATCH:
        raise ToolError(f"Unknown tool {tool_name!r}")

    fn = TOOL_DISPATCH[tool_name]
    # policy_checker only makes sense for create_ad + rewrite_creative
    if tool_name in ("create_ad", "rewrite_creative"):
        return fn(state, day, **args, policy_checker=policy_checker)
    return fn(state, day, **args)
