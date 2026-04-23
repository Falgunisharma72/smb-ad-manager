"""SMB profiles + industry benchmarks for the environment.

Industry benchmark numbers are calibrated from publicly available sources
(WordStream industry CTR/conversion benchmarks, Meta Ads Library spot-checks).
They are realistic but simplified — good enough to make the simulation
defensible without claiming to replicate Meta's exact physics.

Public references used:
    - WordStream 2024-2025 industry benchmarks:
      https://www.wordstream.com/blog/ws/2024/03/05/facebook-advertising-benchmarks
    - Meta Ads Library (sampled reach / impression data across verticals)
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .models import Industry, Objective, SMBProfile


@dataclass(frozen=True)
class IndustryBenchmark:
    """Realistic ad-performance distribution for one vertical.

    All values are per-impression probabilities or INR amounts.
    """
    baseline_ctr: float           # click-through rate (0.0 - 1.0)
    baseline_conv_rate: float     # of clicks, fraction that convert
    avg_order_value_inr: float    # for conversion-based revenue
    cpm_inr: float                # cost per 1000 impressions (audience-neutral)
    ctr_variance: float           # sigma for per-ad variance
    competitive_pressure: float   # 0.5 = easy market, 1.5 = hard market


# Calibrated to WordStream benchmarks + Indian-market CPM rates
# (Indian CPMs are ~5-8× lower than US; values below reflect Indian reality)
INDUSTRY_BENCHMARKS: dict[Industry, IndustryBenchmark] = {
    Industry.SKINCARE: IndustryBenchmark(
        baseline_ctr=0.014,          # 1.4% CTR for beauty/skincare
        baseline_conv_rate=0.025,    # 2.5% of clickers convert
        avg_order_value_inr=850.0,   # typical skincare SKU price
        cpm_inr=95.0,                # ~₹95 per 1000 impressions, competitive vertical
        ctr_variance=0.004,
        competitive_pressure=1.1,
    ),
    Industry.FOOD_DELIVERY: IndustryBenchmark(
        baseline_ctr=0.010,          # 1.0% CTR, a crowded category
        baseline_conv_rate=0.035,    # higher conv rate (impulse + repeat)
        avg_order_value_inr=420.0,   # typical first-order AOV
        cpm_inr=75.0,                # cheaper, broader reach
        ctr_variance=0.003,
        competitive_pressure=1.3,    # Zomato/Swiggy dominate inventory
    ),
    Industry.FITNESS_APPS: IndustryBenchmark(
        baseline_ctr=0.016,          # 1.6% CTR, motivated audience
        baseline_conv_rate=0.018,    # app install = conv; lower than purchase
        avg_order_value_inr=250.0,   # subscription first-month value
        cpm_inr=85.0,
        ctr_variance=0.005,
        competitive_pressure=1.0,
    ),
}


# ─── 10 realistic SMB profiles ───────────────────────────────────────────
# These cover a range of budgets (₹5K - ₹50K/month) and goals.

SMB_PROFILES: list[SMBProfile] = [
    SMBProfile(
        name="Priya's Handmade Candles",
        industry=Industry.SKINCARE,  # natural products — closest vertical
        location="Mumbai, India",
        monthly_budget_inr=10000.0,
        goal="online_sales",
        description="Handmade soy candles, natural scents, targeting urban millennials who care about clean products.",
    ),
    SMBProfile(
        name="Glow Naturally Skincare",
        industry=Industry.SKINCARE,
        location="Bangalore, India",
        monthly_budget_inr=25000.0,
        goal="online_sales",
        description="Ayurveda-inspired skincare line, 12 SKUs. Targeting women 25-45 interested in natural beauty.",
    ),
    SMBProfile(
        name="Serum Story",
        industry=Industry.SKINCARE,
        location="Delhi, India",
        monthly_budget_inr=8000.0,
        goal="brand_awareness",
        description="New D2C vitamin-C serum brand. Want first 1000 customers; brand still unknown.",
    ),
    SMBProfile(
        name="Thali Express",
        industry=Industry.FOOD_DELIVERY,
        location="Pune, India",
        monthly_budget_inr=15000.0,
        goal="online_sales",
        description="Home-style North Indian thalis, lunch + dinner delivery. 6-suburb coverage.",
    ),
    SMBProfile(
        name="Hyderabad Biryani Co.",
        industry=Industry.FOOD_DELIVERY,
        location="Hyderabad, India",
        monthly_budget_inr=30000.0,
        goal="online_sales",
        description="Authentic dum biryani, expanding from 1 to 4 cloud-kitchen locations this quarter.",
    ),
    SMBProfile(
        name="VeggieBowl",
        industry=Industry.FOOD_DELIVERY,
        location="Bangalore, India",
        monthly_budget_inr=5000.0,
        goal="lead_gen",
        description="Sustainable vegan meal bowls. Launching Monday; need pre-orders to validate demand.",
    ),
    SMBProfile(
        name="FitTrack India",
        industry=Industry.FITNESS_APPS,
        location="Gurgaon, India",
        monthly_budget_inr=20000.0,
        goal="app_installs",
        description="Fitness tracker app focused on Indian body types + Desi exercise library.",
    ),
    SMBProfile(
        name="YogaFlow",
        industry=Industry.FITNESS_APPS,
        location="Rishikesh, India",
        monthly_budget_inr=12000.0,
        goal="app_installs",
        description="Live-streamed yoga classes from Rishikesh. Free tier + premium subscription model.",
    ),
    SMBProfile(
        name="RunWell",
        industry=Industry.FITNESS_APPS,
        location="Mumbai, India",
        monthly_budget_inr=50000.0,
        goal="app_installs",
        description="Marathon training app. Large budget, series-A funded, aggressive user acquisition.",
    ),
    SMBProfile(
        name="Herbal Healing Co.",
        industry=Industry.SKINCARE,
        location="Chennai, India",
        monthly_budget_inr=18000.0,
        goal="online_sales",
        description="Herbal remedies and haircare oils, traditional formulations, artisanal positioning.",
    ),
]


def get_smb_profile(index: int | None = None, *, seed: int | None = None) -> SMBProfile:
    """Return one SMB profile, by index or deterministic-random by seed."""
    if index is not None:
        return SMBProfile(**SMB_PROFILES[index].model_dump())
    rng = random.Random(seed)
    return SMBProfile(**rng.choice(SMB_PROFILES).model_dump())


def get_industry_benchmark(industry: Industry) -> IndustryBenchmark:
    """Look up benchmarks for a given vertical."""
    return INDUSTRY_BENCHMARKS[industry]
