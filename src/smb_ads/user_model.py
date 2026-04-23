"""Simulated user-response model.

After a day of campaign activity, this module computes realistic
(impressions, clicks, conversions, revenue) given the ad + audience + budget.

It's deliberately simple — a handful of multiplicative modifiers on the
industry baseline. Just enough to:
  - reward good creative quality,
  - punish ad fatigue,
  - reward audience match,
  - and provide ROAS signal the RL agent can optimize.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .models import Ad, Creative, Industry, Metrics
from .scenarios import IndustryBenchmark, get_industry_benchmark


@dataclass
class CreativeQualitySignals:
    """Heuristic signals used to modify a creative's CTR.

    Not a deep NLP analysis — just a few legible rules that reward well-crafted ads.
    """
    headline_length_ok: bool       # 15-80 chars is a sweet spot
    body_has_cta_language: bool    # "shop now", "try today", etc.
    body_length_ok: bool           # 40-200 chars readable
    mentions_pain_point: bool      # refers to an audience problem
    headline_has_numbers: bool     # "3 ingredients", "in 7 days"

    @property
    def modifier(self) -> float:
        """Multiplier on baseline CTR: 0.5× (bad) to 1.6× (strong)."""
        score = sum([
            self.headline_length_ok,
            self.body_has_cta_language,
            self.body_length_ok,
            self.mentions_pain_point,
            self.headline_has_numbers,
        ])
        # 5 signals: 0 -> 0.5×, 5 -> 1.6×
        return 0.5 + (score / 5.0) * 1.1


# Keywords that bump CTR — rough but debuggable
_CTA_PHRASES = {
    "shop now", "try today", "get yours", "buy now", "order now",
    "sign up", "install now", "get started", "learn more", "book now",
}

_PAIN_POINT_PHRASES = {
    # skincare pain points
    "dry skin", "breakouts", "oily skin", "dull", "wrinkles", "fine lines",
    # food-delivery pain points
    "hungry", "no time", "too busy", "work from home", "late night",
    # fitness pain points
    "beginner", "no gym", "busy schedule", "lose weight", "strength",
}


def _analyze_creative(creative: Creative) -> CreativeQualitySignals:
    """Cheap analysis of creative text — produces CTR modifier signals."""
    h = creative.headline.lower()
    b = creative.body.lower()
    return CreativeQualitySignals(
        headline_length_ok=15 <= len(creative.headline) <= 80,
        body_has_cta_language=any(p in b for p in _CTA_PHRASES),
        body_length_ok=40 <= len(creative.body) <= 250,
        mentions_pain_point=any(p in h + " " + b for p in _PAIN_POINT_PHRASES),
        headline_has_numbers=any(c.isdigit() for c in creative.headline),
    )


def _audience_match_quality(audience_description: str, industry: Industry) -> float:
    """Heuristic: does the audience description mention industry-relevant terms?

    Returns 0.6 (poor) to 1.3 (strong) modifier.
    """
    ad = audience_description.lower()
    relevant_terms = {
        Industry.SKINCARE: ["skin", "beauty", "women", "natural", "ayurveda", "millennial", "gen z"],
        Industry.FOOD_DELIVERY: ["urban", "professional", "busy", "dinner", "lunch", "suburb"],
        Industry.FITNESS_APPS: ["fitness", "gym", "athlete", "runner", "yoga", "active"],
    }
    hits = sum(1 for t in relevant_terms[industry] if t in ad)
    # 0 hits -> 0.6, 3+ hits -> 1.3
    return min(1.3, 0.6 + 0.23 * hits)


def _fatigue_modifier(impressions_to_audience: int, audience_size_est: int = 50_000) -> float:
    """Ad fatigue: when frequency gets high, CTR drops.

    Simplified: drops CTR by 5% per unit of (impressions / audience_size * 10).
    """
    if audience_size_est <= 0:
        return 1.0
    freq = impressions_to_audience / audience_size_est
    # freq 0 -> 1.0, freq 0.2 -> 0.9, freq 1 -> 0.5
    return max(0.5, 1.0 - freq * 0.5)


def simulate_daily_performance(
    *,
    ad: Ad,
    audience_description: str,
    daily_budget_inr: float,
    industry: Industry,
    prior_impressions: int = 0,
    seed: int | None = None,
) -> Metrics:
    """Given one active ad + budget + context, return one day of simulated performance.

    This is the core of the user response model. Called once per simulated day
    per active ad to generate realistic metrics.
    """
    rng = random.Random(seed)
    bench = get_industry_benchmark(industry)

    # How many impressions did the budget buy? Determined by CPM + competitive pressure.
    effective_cpm = bench.cpm_inr * bench.competitive_pressure
    # Small randomness so no two days are identical
    noise = rng.uniform(0.9, 1.1)
    max_impressions = (daily_budget_inr / effective_cpm) * 1000.0 * noise
    impressions = max(0, int(max_impressions))

    # Determine CTR: baseline × creative × audience × fatigue × noise
    creative_signals = _analyze_creative(ad.creative)
    audience_mod = _audience_match_quality(audience_description, industry)
    fatigue_mod = _fatigue_modifier(prior_impressions)
    ctr_noise = rng.gauss(1.0, 0.12)  # daily variance

    ctr = (
        bench.baseline_ctr
        * creative_signals.modifier
        * audience_mod
        * fatigue_mod
        * ctr_noise
    )
    ctr = max(0.001, min(0.08, ctr))  # reasonable bounds

    clicks = int(impressions * ctr)

    # Conversion: smaller noise, less variance
    conv_rate = bench.baseline_conv_rate * rng.gauss(1.0, 0.08)
    conv_rate = max(0.0, min(0.10, conv_rate))
    conversions = int(clicks * conv_rate)

    # Revenue
    aov_noise = rng.gauss(1.0, 0.05)
    revenue = conversions * bench.avg_order_value_inr * aov_noise

    # Spend: usually matches daily_budget (model may underspend if bid too low, ignored here)
    spend = daily_budget_inr * noise

    return Metrics(
        impressions=impressions,
        clicks=clicks,
        conversions=conversions,
        spend_inr=round(spend, 2),
        revenue_inr=round(max(0.0, revenue), 2),
    )


def aggregate_campaign_metrics(
    *,
    ads: list[Ad],
    audience_descriptions: dict[str, str],  # ad_set_id -> audience_description
    daily_budget_inr: float,
    industry: Industry,
    prior_metrics: dict[str, Metrics],  # ad_id -> prior metrics (for fatigue)
    seed: int | None = None,
) -> Metrics:
    """Aggregate one day of performance across all active ads in a campaign."""
    if not ads:
        return Metrics(impressions=0, clicks=0, conversions=0, spend_inr=0.0, revenue_inr=0.0)

    # Split budget evenly across active ads (simplified: real Meta allocates by quality)
    per_ad_budget = daily_budget_inr / len(ads)

    total_imp = 0
    total_clicks = 0
    total_conv = 0
    total_spend = 0.0
    total_revenue = 0.0

    for i, ad in enumerate(ads):
        if ad.status != "active":
            continue
        audience = audience_descriptions.get(ad.ad_set_id, "")
        prior = prior_metrics.get(ad.id, Metrics(
            impressions=0, clicks=0, conversions=0, spend_inr=0.0, revenue_inr=0.0,
        ))
        m = simulate_daily_performance(
            ad=ad,
            audience_description=audience,
            daily_budget_inr=per_ad_budget,
            industry=industry,
            prior_impressions=prior.impressions,
            seed=(seed or 0) * 100 + i if seed is not None else None,
        )
        total_imp += m.impressions
        total_clicks += m.clicks
        total_conv += m.conversions
        total_spend += m.spend_inr
        total_revenue += m.revenue_inr

    return Metrics(
        impressions=total_imp,
        clicks=total_clicks,
        conversions=total_conv,
        spend_inr=round(total_spend, 2),
        revenue_inr=round(total_revenue, 2),
    )
