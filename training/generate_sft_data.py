"""Generate 100 supervised-fine-tuning examples for the Ad Manager agent.

Each example is a (system_prompt, user_prompt, assistant_response) triple where
the assistant response is an 'optimal' action chosen by heuristics, given the
observation.

We use this to warm-start the Llama model before GRPO — guarantees it knows the
JSON format and has seen good action patterns, so RL doesn't have to learn
both simultaneously.

Output: training/sft_data.jsonl  (one JSON object per line, chat-format)

Usage:
    python training/generate_sft_data.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from src.smb_ads.agent import SYSTEM_PROMPT, AdManagerAgent
from src.smb_ads.env import Env
from src.smb_ads.models import Action, ActionType, Observation


# ─── Heuristic policy: given an Observation, what's the "right" action? ──
# These rules implement the behavior we want the model to imitate.

def heuristic_best_action(obs: Observation) -> Action:
    """A hand-crafted 'good' policy we want the small LLM to imitate during SFT.

    Strategy (priority order):
      1. Any ad rejected → rewrite it with a safer creative
      2. We have active campaigns but never fetched metrics → get_metrics
      3. Campaign ROAS > 3.0 → scale budget up 50%
      4. Campaign ROAS < 0.5 (and at least 500 impressions) → pause campaign
      5. Policy drift event occurred and we have skincare ads → rewrite to add disclaimer
      6. Otherwise → noop (wait and observe)
    """
    # Find any rejected ads
    rejected_ads = [a for a in obs.active_ads if a.status == "rejected"]
    if rejected_ads:
        ad = rejected_ads[0]
        safe_creative = _make_safe_creative(ad.creative.headline, ad.creative.body)
        return Action(
            tool=ActionType.REWRITE_CREATIVE,
            args={"ad_id": ad.id, "new_creative": safe_creative},
            reasoning=f"Ad {ad.id} was rejected. Rewriting with policy-compliant language.",
        )

    # Active campaigns but no metrics in the observation → fetch them
    if obs.active_campaigns and not obs.latest_metrics:
        cid = obs.active_campaigns[0].id
        return Action(
            tool=ActionType.GET_METRICS,
            args={"campaign_id": cid},
            reasoning="No metrics seen yet — fetching to inform next action.",
        )

    # Check for winning campaigns (ROAS > 3.0)
    for cid, m in obs.latest_metrics.items():
        if m.roas >= 3.0 and m.spend_inr > 0:
            campaign = next((c for c in obs.active_campaigns if c.id == cid), None)
            if campaign:
                new_budget = round(campaign.daily_budget_inr * 1.5, 2)
                # But don't exceed 1/7 of remaining budget in a single day
                cap = obs.total_budget_remaining_inr / 7.0
                new_budget = min(new_budget, cap)
                return Action(
                    tool=ActionType.UPDATE_BUDGET,
                    args={"campaign_id": cid, "new_daily_budget_inr": new_budget},
                    reasoning=f"Campaign {cid} has ROAS {m.roas:.2f}x — scaling budget.",
                )

    # Check for losing campaigns (ROAS < 0.5 with real data)
    for cid, m in obs.latest_metrics.items():
        if m.roas < 0.5 and m.impressions >= 500:
            return Action(
                tool=ActionType.PAUSE_AD,
                args={"campaign_id": cid},
                reasoning=f"Campaign {cid} has poor ROAS {m.roas:.2f}x — pausing to prevent further loss.",
            )

    # Check for policy drift impact on skincare ads
    policy_update_mentioned = any("POLICY UPDATE" in e for e in obs.recent_events)
    if policy_update_mentioned:
        # If we have skincare-adjacent ads, proactively rewrite one with disclaimer
        for ad in obs.active_ads:
            if ad.status == "active" and _is_skincare_adjacent(ad.creative.body):
                rewritten = _add_disclaimer_to_creative(ad.creative)
                return Action(
                    tool=ActionType.REWRITE_CREATIVE,
                    args={"ad_id": ad.id, "new_creative": rewritten},
                    reasoning="Policy update requires disclaimer — proactively rewriting skincare ad.",
                )

    # No metrics after first step → fetch them (grounds reasoning, prevents H3 hallucination)
    active_campaign_ids = [c.id for c in obs.active_campaigns]
    if active_campaign_ids and obs.step > 0 and not obs.latest_metrics:
        return Action(
            tool=ActionType.GET_METRICS,
            args={"campaign_id": active_campaign_ids[0]},
            reasoning="Refreshing campaign metrics before making any budget decision.",
        )

    # Periodically check policy updates (demonstrates the tool exists)
    if obs.step == 1 and any("POLICY UPDATE" in e for e in obs.recent_events):
        return Action(
            tool=ActionType.GET_POLICY_UPDATES,
            args={},
            reasoning="Policy update just fired — fetching current rules to verify compliance.",
        )

    # Default: observe and wait
    return Action(
        tool=ActionType.NOOP,
        args={},
        reasoning="Metrics look stable; waiting another day to gather more signal.",
    )


# ─── Helpers ─────────────────────────────────────────────────────────────

_SKINCARE_TERMS = {"acne", "skin", "glow", "wrinkle", "hair", "oil", "dry", "dandruff"}

def _is_skincare_adjacent(body: str) -> bool:
    b = body.lower()
    return any(t in b for t in _SKINCARE_TERMS)


def _make_safe_creative(orig_headline: str, orig_body: str) -> dict:
    """Produce a toned-down creative that's less likely to hit policies."""
    return {
        "headline": "Discover Our Natural Collection",
        "body": f"Handcrafted formulations, loved by thousands. Shop now for free shipping. *not medical advice",
        "image_description": "gentle product shot",
        "call_to_action": "shop_now",
    }


def _add_disclaimer_to_creative(creative) -> dict:
    """Add the *not medical advice disclaimer without otherwise changing the ad."""
    body = creative.body
    if "*not medical advice" not in body.lower():
        body = body.rstrip(". ") + ". *not medical advice"
    return {
        "headline": creative.headline,
        "body": body,
        "image_description": creative.image_description,
        "call_to_action": creative.call_to_action.value if hasattr(creative.call_to_action, 'value') else creative.call_to_action,
    }


# ─── Data generation loop ───────────────────────────────────────────────

def generate_examples(n_examples: int = 100, seed: int = 0) -> list[dict]:
    """Run heuristic policy across many seeds/tiers to produce diverse examples."""
    rng = random.Random(seed)
    examples: list[dict] = []
    tiers = ["easy", "medium", "hard"]
    # Weight easy tier more heavily since it's the primary training target
    tier_weights = [0.5, 0.3, 0.2]

    # Each episode contributes up to 7 examples (hard tier max).
    # Aim for enough episodes to comfortably exceed n_examples.
    n_episodes = max(40, n_examples // 2)
    generated = 0
    agent_mock = _HeuristicStub()  # used only to format observations uniformly

    for ep in range(n_episodes):
        if generated >= n_examples:
            break
        tier = rng.choices(tiers, weights=tier_weights)[0]
        ep_seed = rng.randint(0, 10_000)

        env = Env(task_id=tier, seed=ep_seed)
        obs = env.reset()

        while generated < n_examples:
            action = heuristic_best_action(obs)
            user_prompt = agent_mock._format_observation(obs)
            assistant_response = json.dumps({
                "tool": action.tool.value,
                "args": action.args,
                "reasoning": action.reasoning,
            }, ensure_ascii=False)

            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response},
                ],
            })
            generated += 1

            step_result = env.step(action)
            obs = step_result.observation
            if step_result.done:
                break

    return examples[:n_examples]


class _HeuristicStub(AdManagerAgent):
    """Sub-class so we can reuse _format_observation without needing an API client."""
    def __init__(self):
        # Bypass parent __init__ — we don't need a real OpenAI client here
        self.max_retries = 0


# ─── Main ────────────────────────────────────────────────────────────────

def main() -> None:
    out = Path("training/sft_data.jsonl")
    out.parent.mkdir(exist_ok=True)

    examples = generate_examples(n_examples=100, seed=42)
    print(f"Generated {len(examples)} SFT examples.")

    # Quick sanity distribution check
    from collections import Counter
    tool_counts = Counter()
    for ex in examples:
        asst = json.loads(ex["messages"][-1]["content"])
        tool_counts[asst["tool"]] += 1
    print("\nAction distribution:")
    for tool, n in sorted(tool_counts.items(), key=lambda x: -x[1]):
        print(f"  {tool:30s} {n}")

    # Write JSONL
    with out.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nSaved: {out}  ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
