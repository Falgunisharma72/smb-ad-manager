"""Agent harness — the tool-calling loop that turns an LLM into an Ad Manager agent.

Design choices:
    - Uses the OpenAI client (works with OpenAI, Groq, HF, anything compatible).
    - Structured output: asks the model to emit JSON conforming to our Action schema.
    - Robust parsing: tries JSON-mode if available; falls back to regex extraction.
    - Retry on malformed output (up to N attempts) — but each failed attempt still
      advances the step count, so format_compliance (r3) punishes bad behavior.

This module is deliberately framework-light (no LangGraph, no LangChain). A simple
while loop is easier to debug, trace, and to generate SFT training data from.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import OpenAI
from pydantic import ValidationError

from .env import Env
from .models import (
    Action,
    ActionType,
    Observation,
    Reward,
    StepResult,
)


# ─── Prompt templates ────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI Ad Manager for a small business on Meta Ads.
At each step, you will see the current state of the business's ad account, and you must decide ONE action to take.

You have access to 9 tools. Emit exactly one as a JSON object:

{
  "tool": "tool_name",
  "args": { ... tool-specific arguments ... },
  "reasoning": "brief explanation (≤200 chars) of why this action"
}

Available tools and their required args:

- create_campaign         args: {objective: "conversions"|"traffic"|"reach"|"app_installs", daily_budget_inr: float}
- create_ad_set           args: {campaign_id: string, audience_description: string}
- create_ad               args: {ad_set_id: string, creative: {headline, body, image_description, call_to_action}}
- get_metrics             args: {campaign_id: string}
- update_budget           args: {campaign_id: string, new_daily_budget_inr: float}
- pause_ad                args: {ad_id: string}  OR  {campaign_id: string}
- get_policy_updates      args: {}
- rewrite_creative        args: {ad_id: string, new_creative: {headline, body, image_description, call_to_action}}
- noop                    args: {}

Your GOAL is to maximize the business's ROAS (revenue / spend) over the episode while:
 - Staying within the total budget.
 - Never violating ad policies (if an ad is rejected, rewrite it — don't re-push the same content).
 - Only citing metrics you actually fetched via get_metrics.
 - Being efficient (fewer tool calls is better).

Output ONLY the JSON object, no other text."""


USER_PROMPT_TEMPLATE = """Current state:

Business: {smb_name} ({industry}) in {location}
Monthly budget: ₹{monthly_budget:,.0f}
Budget remaining: ₹{budget_remaining:,.0f}
Day {day} of {days_total}

Active campaigns:
{campaigns}

Active ads:
{ads}

Latest metrics per campaign:
{metrics}

Recent events:
{events}

Active policies:
{policies}

What's your next action? Emit a JSON object per the system instructions."""


# ─── Data classes ────────────────────────────────────────────────────────

@dataclass
class TurnRecord:
    """One turn of the agent's trajectory — logged for analysis + SFT data gen."""
    step: int
    observation: dict     # serialized Observation
    llm_output_raw: str   # what the LLM returned verbatim
    parsed_action: Optional[dict]  # parsed action, or None if unparseable
    parse_error: Optional[str]
    reward: Optional[dict]  # serialized Reward, or None if env rejected
    env_done: bool


@dataclass
class AgentRunResult:
    """Everything produced by one agent.run() call."""
    task_id: str
    total_reward: float
    steps: int
    trajectory: list[TurnRecord] = field(default_factory=list)
    env_final_state: dict = field(default_factory=dict)

    def reward_by_component(self) -> dict:
        """Aggregate the 5 reward components across all steps."""
        totals = {
            "r1_roas_improvement": 0.0,
            "r2_policy_compliance": 0,
            "r3_format_compliance": 0,
            "r4_budget_discipline": 0,
            "r5_no_cheating": 0,
        }
        n = 0
        for t in self.trajectory:
            if not t.reward:
                continue
            bd = t.reward.get("breakdown", {})
            for k in totals:
                totals[k] += bd.get(k, 0)
            n += 1
        if n > 0:
            totals["r1_roas_improvement"] /= n  # average, not sum
        return totals


# ─── Agent class ─────────────────────────────────────────────────────────

class AdManagerAgent:
    """LLM-driven Ad Manager agent. Runs a full episode against a local Env."""

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        max_retries_per_step: int = 2,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ):
        self.client = client
        self.model_name = model_name
        self.max_retries = max_retries_per_step
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ────────────────────────────────────────────────────────────────────

    def run(self, env: Env, *, max_env_steps: int = 10) -> AgentRunResult:
        """Run one full episode. Returns trajectory + reward breakdown."""
        obs = env.reset()
        traj: list[TurnRecord] = []
        total_reward = 0.0
        steps = 0

        while steps < max_env_steps:
            steps += 1

            user_prompt = self._format_observation(obs)

            # Try to get a valid action (with a few retries on parse failures)
            parsed_action: Optional[Action] = None
            parse_error: Optional[str] = None
            llm_raw = ""
            for attempt in range(self.max_retries + 1):
                llm_raw = self._call_llm(user_prompt)
                parsed_action, parse_error = self._parse_action(llm_raw)
                if parsed_action is not None:
                    break
                # Retry with explicit correction hint appended
                user_prompt = (
                    user_prompt
                    + f"\n\nYour previous output was unparseable: {parse_error}\n"
                    + "Emit ONLY a valid JSON object as per the system instructions."
                )

            # If we still don't have a valid action, emit a NOOP.
            # The env will still accept it but r3 (format_compliance) hit is already
            # baked into the parse_error — we record it in the trajectory.
            if parsed_action is None:
                parsed_action = Action(
                    tool=ActionType.NOOP,
                    args={},
                    reasoning=f"[agent couldn't parse own output after {self.max_retries+1} tries]",
                )

            # Step the env
            step_result: StepResult = env.step(parsed_action)
            total_reward += step_result.reward.score

            # Record the turn
            traj.append(TurnRecord(
                step=steps,
                observation=obs.model_dump(mode="json"),
                llm_output_raw=llm_raw,
                parsed_action=parsed_action.model_dump(mode="json") if parsed_action else None,
                parse_error=parse_error,
                reward=step_result.reward.model_dump(mode="json"),
                env_done=step_result.done,
            ))

            obs = step_result.observation
            if step_result.done:
                break

        return AgentRunResult(
            task_id=env._task_id,
            total_reward=total_reward,
            steps=steps,
            trajectory=traj,
            env_final_state=env.state(),
        )

    # ─── Internals ───────────────────────────────────────────────────────

    def _format_observation(self, obs: Observation) -> str:
        """Turn an Observation into a compact prompt body."""
        smb = obs.smb_profile

        campaigns_txt = "\n".join(
            f"  - {c.id}: obj={c.objective.value}, daily=₹{c.daily_budget_inr:.0f}, status={c.status}"
            for c in obs.active_campaigns
        ) or "  (none)"

        ads_txt = "\n".join(
            f'  - {a.id}: ad_set={a.ad_set_id}, status={a.status}, headline="{a.creative.headline}"'
            + (f" [rejected: {a.rejection_reason}]" if a.rejection_reason else "")
            for a in obs.active_ads
        ) or "  (none)"

        metrics_txt = "\n".join(
            f"  - {cid}: imp={m.impressions}, clicks={m.clicks}, conv={m.conversions}, "
            f"spend=₹{m.spend_inr:.0f}, revenue=₹{m.revenue_inr:.0f}, ROAS={m.roas:.2f}x"
            for cid, m in obs.latest_metrics.items()
        ) or "  (no metrics fetched this run — use get_metrics to retrieve)"

        events_txt = "\n".join(f"  - {e}" for e in obs.recent_events) or "  (none)"

        policies_txt = "\n".join(
            f"  - {p.id}: {p.name}" for p in obs.active_policies
        ) or "  (none active)"

        return USER_PROMPT_TEMPLATE.format(
            smb_name=smb.name,
            industry=smb.industry.value,
            location=smb.location,
            monthly_budget=smb.monthly_budget_inr,
            budget_remaining=obs.total_budget_remaining_inr,
            day=obs.day,
            days_total=obs.days_total,
            campaigns=campaigns_txt,
            ads=ads_txt,
            metrics=metrics_txt,
            events=events_txt,
            policies=policies_txt,
        )

    def _call_llm(self, user_prompt: str) -> str:
        """One call to the configured OpenAI-compatible endpoint."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            # Try with JSON mode; many providers (Groq/OpenAI) support it
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception:
            # Fall back without JSON mode (HF Inference API, some others)
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        return (resp.choices[0].message.content or "").strip()

    def _parse_action(self, raw: str) -> tuple[Optional[Action], Optional[str]]:
        """Parse an LLM response into a valid Action, or return (None, error_msg)."""
        if not raw:
            return None, "empty response"

        # Strip code fences if present (common LLM habit)
        cleaned = _strip_code_fence(raw)

        # Try direct parse first
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            # Try to find the first {...} block
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if match is None:
                return None, f"no JSON object in response: {e}"
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError as e2:
                return None, f"malformed JSON: {e2}"

        try:
            action = Action(**data)
        except ValidationError as e:
            return None, f"schema validation failed: {e.errors()[0]['msg']}"
        except TypeError as e:
            return None, f"wrong types: {e}"

        return action, None


# ─── Utilities ───────────────────────────────────────────────────────────

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)

def _strip_code_fence(s: str) -> str:
    """Remove ```json ... ``` wrappers if the model added them."""
    s = s.strip()
    m = _CODE_FENCE_RE.match(s)
    return m.group(1) if m else s
