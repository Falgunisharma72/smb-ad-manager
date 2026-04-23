"""Baseline inference — the hackathon-mandated reference implementation.

Reads env vars per hackathon spec:
    OPENAI_API_KEY  — your API key (Groq/OpenAI/etc)
    API_BASE_URL    — OpenAI-compatible endpoint base URL (optional for OpenAI default)
    MODEL_NAME      — model identifier (default: llama-3.3-70b-versatile)
    HF_TOKEN        — HuggingFace token (optional, not used by this baseline)
    TASK_ID         — 'easy' | 'medium' | 'hard' (default: easy)

Usage:
    python inference.py

This drives a real AdManagerAgent through one episode and prints the full
trajectory + aggregated reward breakdown.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.smb_ads.agent import AdManagerAgent
from src.smb_ads.env import Env


def main() -> int:
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("API_BASE_URL") or None
    model = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
    task = os.environ.get("TASK_ID", "easy")
    seed_raw = os.environ.get("SEED", "42")
    try:
        seed = int(seed_raw)
    except (TypeError, ValueError):
        seed = None

    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set. See .env.example.", file=sys.stderr)
        return 1

    print(f"[inference] model={model}")
    print(f"[inference] endpoint={base_url or 'default (OpenAI)'}")
    print(f"[inference] task={task}")
    print(f"[inference] seed={seed}")
    print()

    client = OpenAI(api_key=api_key, base_url=base_url)

    agent = AdManagerAgent(
        client=client,
        model_name=model,
        max_retries_per_step=2,
        temperature=0.3,
        max_tokens=512,
    )

    env = Env(task_id=task, seed=seed)
    result = agent.run(env, max_env_steps=10)

    # Print a concise trajectory report
    print()
    print(f"=== Trajectory ({result.steps} steps, total reward {result.total_reward:.3f}) ===")
    for t in result.trajectory:
        tool = (t.parsed_action or {}).get("tool", "n/a")
        reasoning = (t.parsed_action or {}).get("reasoning", "")[:80]
        score = (t.reward or {}).get("score", 0.0)
        print(f"  step {t.step}: tool={tool:<20s} score={score:.3f}  | {reasoning}")

    # Reward breakdown
    print()
    print("=== Aggregate reward breakdown ===")
    agg = result.reward_by_component()
    for k, v in agg.items():
        print(f"  {k:30s} {v}")

    # Save full trajectory JSON for later inspection / SFT data prep
    out_path = Path("logs/last_trajectory.json")
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        json.dump({
            "task_id": result.task_id,
            "total_reward": result.total_reward,
            "steps": result.steps,
            "reward_breakdown": agg,
            "trajectory": [t.__dict__ for t in result.trajectory],
            "final_state": result.env_final_state,
        }, f, indent=2, default=str)
    print(f"\n[inference] full trajectory saved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
