"""Baseline agent — the hackathon-mandated reference implementation.

Reads env vars per hackathon spec:
    OPENAI_API_KEY  — your API key (Groq/OpenAI/etc)
    API_BASE_URL    — OpenAI-compatible endpoint base URL (optional for OpenAI default)
    MODEL_NAME      — model identifier (default: llama-3.3-70b-versatile)
    HF_TOKEN        — HuggingFace token (optional)
    TASK_ID         — 'easy' | 'medium' | 'hard' (default: easy)

Currently a SKELETON (Hour 0-4). The real LangGraph tool-calling loop lands in
Hour 10-11. This version just validates the plumbing.
"""
from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from src.smb_ads.env import Env
from src.smb_ads.models import Action, ActionType


def main() -> None:
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("API_BASE_URL") or None
    model = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
    task = os.environ.get("TASK_ID", "easy")

    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set. See .env.example.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=base_url)

    print(f"[inference] model={model} task={task} base_url={base_url or 'default'}")

    env = Env(task_id=task)
    obs = env.reset()

    max_steps = {"easy": 1, "medium": 3, "hard": 7}[task]
    for step_idx in range(max_steps):
        # Placeholder for real agent logic — LangGraph loop lands Hour 10-11.
        # For now: emit a noop action to verify the plumbing.
        action = Action(
            tool=ActionType.NOOP,
            args={},
            reasoning="baseline placeholder — no model call yet; validates env wiring",
        )

        result = env.step(action)
        print(f"[inference] step={step_idx} reward={result.reward.score:.3f} done={result.done}")

        if result.done:
            break

    print("[inference] baseline run complete.")


if __name__ == "__main__":
    main()
