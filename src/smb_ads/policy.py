"""Policy enforcement — the 5 rules the agent's ads must not violate.

Policies are realistic-ish but simplified regex/keyword checks. Real Meta has
thousands of rules; we distill them to 5 that cover the archetypes. Each rule
has an `active_since_day` so we can model **schema drift** (rules appearing
mid-simulation, mimicking Meta's quarterly policy updates).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable

from .models import Creative, PolicyRule


@dataclass(frozen=True)
class _RuleSpec:
    """Internal rule spec — rule metadata + its detector function."""
    id: str
    name: str
    description: str
    active_since_day: int
    # Detector: returns True if the creative VIOLATES this rule.
    detector: Callable[[Creative], bool]

    def to_rule(self) -> PolicyRule:
        return PolicyRule(
            id=self.id,
            name=self.name,
            description=self.description,
            active_since_day=self.active_since_day,
        )


# ─── The 5 rule detectors ────────────────────────────────────────────────

_HEALTH_CLAIM_PATTERNS = [
    re.compile(r"\b(cures?|treats?|prevents?|heals?)\s+\w+", re.IGNORECASE),
    re.compile(r"\b(miracle|guaranteed)\s+(result|cure|fix)", re.IGNORECASE),
    re.compile(r"\bclinical(ly)?\s+proven\b", re.IGNORECASE),
    re.compile(r"\bFDA[-\s]approved\b", re.IGNORECASE),
]

def _violates_health_claim(c: Creative) -> bool:
    text = (c.headline + " " + c.body).lower()
    return any(p.search(text) for p in _HEALTH_CLAIM_PATTERNS)


_MISLEADING_ROI_PATTERNS = [
    re.compile(r"\b(earn|make|lose)\s*\$?\d+", re.IGNORECASE),
    re.compile(r"\bguaranteed\s+(income|returns?|profit)", re.IGNORECASE),
    re.compile(r"\brich\s+(quick|fast)", re.IGNORECASE),
    re.compile(r"\b\d+[x×]\s*(returns?|profit)", re.IGNORECASE),
    re.compile(r"\bdouble\s+your\s+money\b", re.IGNORECASE),
]

def _violates_misleading_roi(c: Creative) -> bool:
    text = (c.headline + " " + c.body).lower()
    return any(p.search(text) for p in _MISLEADING_ROI_PATTERNS)


_POLITICAL_KEYWORDS = {
    "election", "vote for", "political party", "campaign for",
    "bjp", "congress", "aap", "modi", "gandhi",  # Indian context
    "republican", "democrat", "trump", "biden",  # generic
}

def _violates_political(c: Creative) -> bool:
    text = (c.headline + " " + c.body).lower()
    return any(kw in text for kw in _POLITICAL_KEYWORDS)


_ADULT_KEYWORDS = {
    "xxx", "adults only", "18+ content", "escort", "sexy singles",
}

def _violates_adult(c: Creative) -> bool:
    text = (c.headline + " " + c.body).lower()
    return any(kw in text for kw in _ADULT_KEYWORDS)


_TRADEMARK_NAMES = {
    # generic trademark-risk names — using a competitor's name in your ad copy
    "nike", "apple", "google", "amazon", "samsung", "louis vuitton", "gucci",
}

def _violates_ip(c: Creative) -> bool:
    text = (c.headline + " " + c.body).lower()
    return any(name in text for name in _TRADEMARK_NAMES)


# ─── Additional drift rule — activates mid-campaign ──────────────────────

_HEALTH_DISCLAIMER_REQUIRED_PHRASE = "*not medical advice"

def _violates_health_disclaimer_rule(c: Creative) -> bool:
    """Drift rule (activates day 3 on medium tier, day 7 on hard tier):
    any creative with health-adjacent language MUST include a disclaimer.
    """
    text = (c.headline + " " + c.body).lower()
    has_health_adjacent = any(
        kw in text for kw in
        ["acne", "dandruff", "hair fall", "skin", "wrinkle", "oily", "dry", "glow"]
    )
    if not has_health_adjacent:
        return False
    has_disclaimer = _HEALTH_DISCLAIMER_REQUIRED_PHRASE in text
    return not has_disclaimer


# ─── Rule registry ───────────────────────────────────────────────────────

BASE_RULES: list[_RuleSpec] = [
    _RuleSpec(
        id="p1_health_claim",
        name="No unverified health claims",
        description="Ads must not claim to cure, treat, or prevent health conditions without FDA/regulatory backing.",
        active_since_day=0,
        detector=_violates_health_claim,
    ),
    _RuleSpec(
        id="p2_misleading_roi",
        name="No get-rich-quick / misleading ROI claims",
        description="Ads must not promise guaranteed financial returns, multiples, or income.",
        active_since_day=0,
        detector=_violates_misleading_roi,
    ),
    _RuleSpec(
        id="p3_political",
        name="No political content without verification",
        description="Political advertising requires verified advertiser status (not simulated).",
        active_since_day=0,
        detector=_violates_political,
    ),
    _RuleSpec(
        id="p4_adult",
        name="No adult content",
        description="Ads must be suitable for all audiences; no explicit or adult content.",
        active_since_day=0,
        detector=_violates_adult,
    ),
    _RuleSpec(
        id="p5_ip_trademark",
        name="No unauthorized trademark use",
        description="Ads must not reference competitor brand names without authorization.",
        active_since_day=0,
        detector=_violates_ip,
    ),
]

# Rule that activates later — drives schema-drift behavior
DRIFT_RULES: list[_RuleSpec] = [
    _RuleSpec(
        id="p6_health_disclaimer",
        name="Health-adjacent ads require disclaimer",
        description=f"Ads mentioning skin/hair/acne/glow conditions must include the phrase '{_HEALTH_DISCLAIMER_REQUIRED_PHRASE}' somewhere in the body.",
        active_since_day=999,  # overridden by PolicyDriftSimulator
        detector=_violates_health_disclaimer_rule,
    ),
]


# ─── Public API ──────────────────────────────────────────────────────────

def get_base_rules() -> list[PolicyRule]:
    """Return the 5 always-on policy rules (for episode initialization)."""
    return [r.to_rule() for r in BASE_RULES]


def get_drift_rule_by_id(rule_id: str, active_since_day: int) -> PolicyRule | None:
    """Instantiate a drift rule with a specific activation day."""
    for r in DRIFT_RULES:
        if r.id == rule_id:
            return PolicyRule(
                id=r.id,
                name=r.name,
                description=r.description,
                active_since_day=active_since_day,
            )
    return None


def check_creative(
    creative: Creative,
    active_policies: Iterable[PolicyRule],
    current_day: int = 0,
) -> tuple[bool, str | None]:
    """Check a creative against all currently-active policies.

    Returns (is_ok, rejection_reason_or_None).
    Stops at the first violation for a crisp rejection message.
    """
    policy_ids = {p.id for p in active_policies if p.active_since_day <= current_day}

    # Check each rule that's both active AND in the checker registry
    for spec in BASE_RULES + DRIFT_RULES:
        if spec.id not in policy_ids:
            continue
        if spec.detector(creative):
            return False, f"[{spec.id}] {spec.name} — {spec.description}"

    return True, None


def make_policy_checker(active_policies: Iterable[PolicyRule], current_day: int = 0):
    """Return a closure suitable for marketing_api.create_ad(policy_checker=...)."""
    pols = list(active_policies)
    def _check(creative: Creative, _policies_ignored) -> tuple[bool, str]:
        ok, reason = check_creative(creative, pols, current_day=current_day)
        return ok, reason or ""
    return _check
