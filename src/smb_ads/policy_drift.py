"""Policy drift simulator — makes policies change mid-episode.

This is the feature that qualifies us for the Patronus bonus track ("schema
drift in consumer workflows"). Rules appear or change over the simulation
timeline, modeling how Meta updates ad policies ~quarterly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .models import PolicyRule
from .policy import get_base_rules, get_drift_rule_by_id


@dataclass(frozen=True)
class DriftEvent:
    """One scheduled policy change."""
    day: int
    rule_id: str  # must match a rule in policy.DRIFT_RULES
    kind: Literal["activate"] = "activate"  # could add "deactivate" / "modify" later


# Drift schedules per task tier
DRIFT_SCHEDULES: dict[str, list[DriftEvent]] = {
    "easy": [],  # no drift on easy
    "medium": [
        DriftEvent(day=2, rule_id="p6_health_disclaimer", kind="activate"),
    ],
    "hard": [
        DriftEvent(day=3, rule_id="p6_health_disclaimer", kind="activate"),
        # room for more drift events on hard tier as we expand the rule library
    ],
}


def initial_policies(task_id: str) -> list[PolicyRule]:
    """The policies active at day 0, before any drift."""
    return get_base_rules()


def apply_drift(
    current_policies: dict[str, PolicyRule],
    task_id: str,
    current_day: int,
) -> tuple[dict[str, PolicyRule], list[str]]:
    """Apply any scheduled drift events at or before current_day.

    Returns (updated_policies_dict, list_of_event_descriptions_for_observation).
    """
    schedule = DRIFT_SCHEDULES.get(task_id, [])
    events_applied: list[str] = []

    for event in schedule:
        if event.day > current_day:
            continue
        if event.rule_id in current_policies:
            continue  # already applied
        if event.kind == "activate":
            new_rule = get_drift_rule_by_id(event.rule_id, active_since_day=event.day)
            if new_rule is not None:
                current_policies[new_rule.id] = new_rule
                events_applied.append(
                    f"POLICY UPDATE (day {event.day}): {new_rule.name} — {new_rule.description}"
                )

    return current_policies, events_applied
