"""Unit tests for the 8 mock Meta Marketing API endpoints."""
from __future__ import annotations

import pytest

from src.smb_ads import marketing_api
from src.smb_ads.marketing_api import ToolError
from src.smb_ads.models import Industry, SMBProfile
from src.smb_ads.state import AccountState


@pytest.fixture
def state():
    smb = SMBProfile(
        name="Test Biz",
        industry=Industry.SKINCARE,
        location="Mumbai",
        monthly_budget_inr=10000.0,
        goal="online_sales",
        description="skincare test",
    )
    return AccountState(smb=smb, total_budget_inr=10000.0)


@pytest.fixture
def valid_creative():
    return {
        "headline": "Glow Naturally This Summer",
        "body": "Try our hand-crafted natural skincare. Free shipping across India.",
        "image_description": "warm-lit bathroom shelf with 3 amber glass bottles",
        "call_to_action": "shop_now",
    }


# ─── create_campaign ─────────────────────────────────────────────────────

def test_create_campaign_happy(state):
    r = marketing_api.create_campaign(
        state, day=0, objective="conversions", daily_budget_inr=500.0,
    )
    assert r["campaign_id"].startswith("c")
    assert r["status"] == "active"
    assert len(state.campaigns) == 1


def test_create_campaign_bad_objective(state):
    with pytest.raises(ToolError, match="Invalid objective"):
        marketing_api.create_campaign(
            state, day=0, objective="not_a_thing", daily_budget_inr=500.0,
        )


def test_create_campaign_overspend_blocked(state):
    # Budget remaining is 10000; asking for 20000 daily should fail
    with pytest.raises(ToolError, match="exceeds remaining budget"):
        marketing_api.create_campaign(
            state, day=0, objective="conversions", daily_budget_inr=20000.0,
        )


# ─── create_ad_set ───────────────────────────────────────────────────────

def test_create_ad_set_happy(state):
    cid = marketing_api.create_campaign(
        state, day=0, objective="conversions", daily_budget_inr=500.0,
    )["campaign_id"]
    r = marketing_api.create_ad_set(
        state, day=0, campaign_id=cid, audience_description="women 25-45 skincare India",
    )
    assert r["ad_set_id"].startswith("as")
    assert len(state.ad_sets) == 1


def test_create_ad_set_unknown_campaign(state):
    with pytest.raises(ToolError, match="Unknown campaign_id"):
        marketing_api.create_ad_set(
            state, day=0, campaign_id="cdoesnotexist", audience_description="test audience",
        )


# ─── create_ad ──────────────────────────────────────────────────────────

def test_create_ad_happy(state, valid_creative):
    cid = marketing_api.create_campaign(
        state, day=0, objective="conversions", daily_budget_inr=500.0,
    )["campaign_id"]
    asid = marketing_api.create_ad_set(
        state, day=0, campaign_id=cid, audience_description="women 25-45 skincare",
    )["ad_set_id"]
    r = marketing_api.create_ad(
        state, day=0, ad_set_id=asid, creative=valid_creative, policy_checker=None,
    )
    assert r["ad_id"].startswith("ad")
    assert r["status"] == "active"


def test_create_ad_malformed_creative(state):
    cid = marketing_api.create_campaign(
        state, day=0, objective="conversions", daily_budget_inr=500.0,
    )["campaign_id"]
    asid = marketing_api.create_ad_set(
        state, day=0, campaign_id=cid, audience_description="test audience",
    )["ad_set_id"]
    with pytest.raises(ToolError, match="Invalid creative"):
        marketing_api.create_ad(
            state, day=0, ad_set_id=asid, creative={"missing": "fields"},
        )


def test_create_ad_policy_rejection(state, valid_creative):
    """If the policy_checker returns (False, reason), ad is rejected and state is updated."""
    cid = marketing_api.create_campaign(
        state, day=0, objective="conversions", daily_budget_inr=500.0,
    )["campaign_id"]
    asid = marketing_api.create_ad_set(
        state, day=0, campaign_id=cid, audience_description="test audience",
    )["ad_set_id"]

    def reject_all(creative, policies):
        return False, "Test reject"

    r = marketing_api.create_ad(
        state, day=0, ad_set_id=asid, creative=valid_creative, policy_checker=reject_all,
    )
    assert r["status"] == "rejected"
    assert r["rejection_reason"] == "Test reject"
    assert state.rejected_ads_this_episode == 1


# ─── get_metrics ─────────────────────────────────────────────────────────

def test_get_metrics_zero_when_no_data(state):
    cid = marketing_api.create_campaign(
        state, day=0, objective="conversions", daily_budget_inr=500.0,
    )["campaign_id"]
    r = marketing_api.get_metrics(state, day=0, campaign_id=cid)
    assert r["impressions"] == 0
    assert r["roas"] == 0.0


def test_get_metrics_unknown_campaign(state):
    with pytest.raises(ToolError, match="Unknown campaign_id"):
        marketing_api.get_metrics(state, day=0, campaign_id="bogus")


# ─── update_budget ───────────────────────────────────────────────────────

def test_update_budget_happy(state):
    cid = marketing_api.create_campaign(
        state, day=0, objective="conversions", daily_budget_inr=500.0,
    )["campaign_id"]
    r = marketing_api.update_budget(
        state, day=0, campaign_id=cid, new_daily_budget_inr=800.0,
    )
    assert r["new_budget"] == 800.0
    assert state.campaigns[cid].daily_budget_inr == 800.0


# ─── pause ───────────────────────────────────────────────────────────────

def test_pause_campaign_also_pauses_ads(state, valid_creative):
    cid = marketing_api.create_campaign(
        state, day=0, objective="conversions", daily_budget_inr=500.0,
    )["campaign_id"]
    asid = marketing_api.create_ad_set(
        state, day=0, campaign_id=cid, audience_description="test audience",
    )["ad_set_id"]
    ad_id = marketing_api.create_ad(
        state, day=0, ad_set_id=asid, creative=valid_creative,
    )["ad_id"]

    marketing_api.pause_ad(state, day=0, campaign_id=cid)

    assert state.campaigns[cid].status == "paused"
    assert state.ads[ad_id].status == "paused"


def test_pause_requires_exactly_one_target(state):
    with pytest.raises(ToolError, match="Provide exactly one"):
        marketing_api.pause_ad(state, day=0)
    with pytest.raises(ToolError, match="Provide exactly one"):
        marketing_api.pause_ad(state, day=0, ad_id="x", campaign_id="y")


# ─── policy + rewrite ───────────────────────────────────────────────────

def test_get_policy_updates_empty_initially(state):
    r = marketing_api.get_policy_updates(state, day=0)
    assert r["count"] == 0


def test_rewrite_creative_unblocks_rejected_ad(state, valid_creative):
    cid = marketing_api.create_campaign(
        state, day=0, objective="conversions", daily_budget_inr=500.0,
    )["campaign_id"]
    asid = marketing_api.create_ad_set(
        state, day=0, campaign_id=cid, audience_description="test audience",
    )["ad_set_id"]

    def reject_first_accept_next(creative, policies):
        # Stateful: returns False first time, True after
        if "Test reject" not in getattr(reject_first_accept_next, "seen", set()):
            reject_first_accept_next.seen = {"Test reject"}
            return False, "Test reject"
        return True, ""

    r = marketing_api.create_ad(
        state, day=0, ad_set_id=asid, creative=valid_creative,
        policy_checker=reject_first_accept_next,
    )
    assert r["status"] == "rejected"
    ad_id = r["ad_id"]

    r2 = marketing_api.rewrite_creative(
        state, day=1, ad_id=ad_id,
        new_creative={**valid_creative, "body": "different rewritten body"},
        policy_checker=reject_first_accept_next,
    )
    assert r2["status"] == "active"


# ─── dispatcher ──────────────────────────────────────────────────────────

def test_dispatcher_noop(state):
    r = marketing_api.dispatch("noop", state, day=0, args={})
    assert r["status"] == "noop"


def test_dispatcher_unknown_tool(state):
    with pytest.raises(ToolError, match="Unknown tool"):
        marketing_api.dispatch("not_a_tool", state, day=0, args={})
