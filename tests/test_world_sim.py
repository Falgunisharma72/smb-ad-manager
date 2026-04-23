"""Tests for the world simulation layer: scenarios, user_model, policy, drift."""
from __future__ import annotations

import pytest

from src.smb_ads import policy, policy_drift, scenarios, user_model
from src.smb_ads.models import Ad, Creative, Industry, PolicyRule
from src.smb_ads.scenarios import SMB_PROFILES, get_industry_benchmark


# ─── scenarios ───────────────────────────────────────────────────────────

def test_all_industries_have_benchmarks():
    for ind in Industry:
        b = get_industry_benchmark(ind)
        assert b.baseline_ctr > 0
        assert b.avg_order_value_inr > 0


def test_smb_profiles_are_valid_and_diverse():
    assert len(SMB_PROFILES) >= 10
    industries = {p.industry for p in SMB_PROFILES}
    assert len(industries) == 3  # all 3 verticals covered
    budgets = {p.monthly_budget_inr for p in SMB_PROFILES}
    assert min(budgets) < 10_000 and max(budgets) > 25_000  # budget range


def test_get_smb_profile_by_index():
    p = scenarios.get_smb_profile(index=0)
    assert p.name == SMB_PROFILES[0].name


def test_get_smb_profile_deterministic_with_seed():
    p1 = scenarios.get_smb_profile(seed=42)
    p2 = scenarios.get_smb_profile(seed=42)
    assert p1.name == p2.name


# ─── user_model ──────────────────────────────────────────────────────────

@pytest.fixture
def quality_ad():
    return Ad(
        id="ad001",
        ad_set_id="as001",
        creative=Creative(
            headline="Transform Dry Skin in 7 Days",
            body="Try our 3-ingredient natural serum. Shop now for a glow that lasts. Free shipping.",
            image_description="amber glass bottle on marble",
            call_to_action="shop_now",
        ),
        status="active",
        created_day=0,
    )


@pytest.fixture
def weak_ad():
    return Ad(
        id="ad002",
        ad_set_id="as002",
        creative=Creative(
            headline="Buy stuff",
            body="Product available here.",
            image_description="plain product photo",
            call_to_action="learn_more",
        ),
        status="active",
        created_day=0,
    )


def test_quality_ad_outperforms_weak_ad(quality_ad, weak_ad):
    """A carefully-crafted creative should get higher CTR than a lazy one, consistently."""
    audience = "women 25-45 urban skincare enthusiasts natural beauty"
    q_total = 0
    w_total = 0
    for seed in range(20):  # average over seeds to smooth variance
        q = user_model.simulate_daily_performance(
            ad=quality_ad,
            audience_description=audience,
            daily_budget_inr=500.0,
            industry=Industry.SKINCARE,
            seed=seed,
        )
        w = user_model.simulate_daily_performance(
            ad=weak_ad,
            audience_description=audience,
            daily_budget_inr=500.0,
            industry=Industry.SKINCARE,
            seed=seed,
        )
        q_total += q.clicks
        w_total += w.clicks
    assert q_total > w_total  # quality creative wins


def test_metrics_are_non_negative(quality_ad):
    m = user_model.simulate_daily_performance(
        ad=quality_ad,
        audience_description="urban women",
        daily_budget_inr=100.0,
        industry=Industry.SKINCARE,
        seed=1,
    )
    assert m.impressions >= 0
    assert m.clicks >= 0 and m.clicks <= m.impressions
    assert m.conversions >= 0 and m.conversions <= m.clicks
    assert m.spend_inr >= 0
    assert m.revenue_inr >= 0


def test_deterministic_with_seed(quality_ad):
    m1 = user_model.simulate_daily_performance(
        ad=quality_ad, audience_description="x", daily_budget_inr=500.0,
        industry=Industry.SKINCARE, seed=42,
    )
    m2 = user_model.simulate_daily_performance(
        ad=quality_ad, audience_description="x", daily_budget_inr=500.0,
        industry=Industry.SKINCARE, seed=42,
    )
    assert m1.impressions == m2.impressions
    assert m1.clicks == m2.clicks


def test_zero_budget_yields_zero_metrics(quality_ad):
    m = user_model.simulate_daily_performance(
        ad=quality_ad, audience_description="x", daily_budget_inr=0.0,
        industry=Industry.SKINCARE, seed=0,
    )
    assert m.impressions == 0
    assert m.clicks == 0


# ─── policy ──────────────────────────────────────────────────────────────

@pytest.fixture
def clean_creative():
    return Creative(
        headline="Fresh handmade candles",
        body="Soy wax, natural scents, made in Mumbai. Shop now for cozy winter vibes.",
        image_description="candles on wooden table",
        call_to_action="shop_now",
    )


def test_clean_creative_passes_all_policies(clean_creative):
    ok, reason = policy.check_creative(clean_creative, policy.get_base_rules(), current_day=0)
    assert ok is True
    assert reason is None


def test_health_claim_is_rejected():
    c = Creative(
        headline="Cure your acne in 3 days",
        body="Clinically proven to heal breakouts fast. FDA approved formula.",
        image_description="test",
        call_to_action="shop_now",
    )
    ok, reason = policy.check_creative(c, policy.get_base_rules(), current_day=0)
    assert ok is False
    assert "health claim" in reason.lower()


def test_misleading_roi_is_rejected():
    c = Creative(
        headline="Get rich quick scheme",
        body="Earn $5000 guaranteed returns in one week. Double your money!",
        image_description="stacks of cash",
        call_to_action="learn_more",
    )
    ok, reason = policy.check_creative(c, policy.get_base_rules(), current_day=0)
    assert ok is False
    assert "roi" in reason.lower() or "misleading" in reason.lower()


def test_political_is_rejected():
    c = Creative(
        headline="Vote for our party",
        body="BJP has the best policies this election. Support the campaign.",
        image_description="flag",
        call_to_action="learn_more",
    )
    ok, reason = policy.check_creative(c, policy.get_base_rules(), current_day=0)
    assert ok is False


def test_trademark_is_rejected():
    c = Creative(
        headline="Better than Nike",
        body="Our running gear beats Nike at half the price.",
        image_description="shoes",
        call_to_action="shop_now",
    )
    ok, reason = policy.check_creative(c, policy.get_base_rules(), current_day=0)
    assert ok is False


def test_make_policy_checker_callable(clean_creative):
    checker = policy.make_policy_checker(policy.get_base_rules(), current_day=0)
    ok, reason = checker(clean_creative, None)
    assert ok is True
    assert reason == ""


# ─── policy drift ────────────────────────────────────────────────────────

def test_no_drift_on_easy_tier():
    pols = {p.id: p for p in policy_drift.initial_policies("easy")}
    for day in range(10):
        pols, events = policy_drift.apply_drift(pols, "easy", current_day=day)
        assert events == []
    # Only base 5 rules
    assert len(pols) == 5


def test_drift_activates_on_medium_tier_day_2():
    pols = {p.id: p for p in policy_drift.initial_policies("medium")}
    # Days 0, 1 — no drift yet
    pols, events = policy_drift.apply_drift(pols, "medium", current_day=0)
    assert events == []
    assert "p6_health_disclaimer" not in pols

    # Day 2 — drift fires
    pols, events = policy_drift.apply_drift(pols, "medium", current_day=2)
    assert len(events) == 1
    assert "p6_health_disclaimer" in pols
    assert "POLICY UPDATE" in events[0]


def test_drift_rule_actually_affects_policy_check():
    """Before drift: skincare ad without disclaimer is OK.
    After drift: same ad is rejected."""
    c = Creative(
        headline="Your glow awaits",
        body="Natural serum for dry skin. Shop now.",
        image_description="test",
        call_to_action="shop_now",
    )

    pols_before = policy_drift.initial_policies("medium")
    ok_before, _ = policy.check_creative(c, pols_before, current_day=0)
    assert ok_before is True

    # Apply drift
    pols_dict = {p.id: p for p in pols_before}
    pols_dict, _ = policy_drift.apply_drift(pols_dict, "medium", current_day=2)
    pols_after = list(pols_dict.values())

    ok_after, reason = policy.check_creative(c, pols_after, current_day=2)
    assert ok_after is False
    assert "disclaimer" in reason.lower()


def test_drift_with_disclaimer_passes():
    """After drift, ads WITH the disclaimer pass."""
    c = Creative(
        headline="Your glow awaits",
        body="Natural serum for dry skin. Shop now. *not medical advice",
        image_description="test",
        call_to_action="shop_now",
    )
    pols_dict = {p.id: p for p in policy_drift.initial_policies("medium")}
    pols_dict, _ = policy_drift.apply_drift(pols_dict, "medium", current_day=2)
    ok, _ = policy.check_creative(c, list(pols_dict.values()), current_day=2)
    assert ok is True
