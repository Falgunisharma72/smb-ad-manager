# %% [markdown]
# # GRPO Training for SMB Ad Manager Agent
#
# RL fine-tuning using TRL's GRPOTrainer, starting from the SFT adapter.
# Rewards come from our live HF Space — each generation is replayed against
# the environment, which returns a composite reward + 5 logged components.
#
# **Cite for the pitch:**
# - GRPO (DeepSeek-R1): arxiv 2501.12948
# - 2-GRPO (group size 2): arxiv 2510.00977 — 12.5% rollouts of standard GRPO
#
# **Expected runtime:**
# - Colab L4: ~4-5 hours for 200 steps
# - DGX A100: ~30-45 min
#
# **Output:** `./checkpoints/grpo_final/` (the trained LoRA)

# %%
# Cell 1 — Install deps (Colab)
# !pip install -qqq unsloth trl peft bitsandbytes wandb requests

# %%
# Cell 2 — Config
import os
from pathlib import Path

SFT_CHECKPOINT = "./checkpoints/sft_adapter"
OUTPUT_DIR = "./checkpoints/grpo_final"
SPACE_URL = os.environ.get("SPACE_URL", "https://falgunisharma-smb-ad-manager.hf.space")
MAX_SEQ_LENGTH = 2048
NUM_GENERATIONS = 2         # 2-GRPO per arxiv 2510.00977
TOTAL_STEPS = 200           # reduce to 50 for initial smoke test
LEARNING_RATE = 1e-5
WANDB_PROJECT = "smb-ad-manager"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
os.environ["WANDB_PROJECT"] = WANDB_PROJECT
print(f"Space URL: {SPACE_URL}")
print(f"Training: {TOTAL_STEPS} steps, group size {NUM_GENERATIONS} (2-GRPO)")

# %%
# Cell 3 — Load SFT-warm-started model
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=SFT_CHECKPOINT,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)
FastLanguageModel.for_training(model)

# %%
# Cell 4 — Define the reward function that calls our HF Space
import json
import re
import time
from typing import Any

import requests

# Simple session + retry logic (the Space has cold-start latency)
_SESSION = requests.Session()

def _call_space(path: str, payload: dict) -> dict | None:
    for attempt in range(3):
        try:
            r = _SESSION.post(f"{SPACE_URL}{path}", json=payload, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[space] attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    return None


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

def _extract_json(text: str) -> dict | None:
    m = _JSON_OBJ_RE.search(text.strip())
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def env_reward(prompts, completions, **kwargs) -> list[float]:
    """Reward function called by GRPOTrainer.

    For each (prompt, completion) pair, we:
      1. Reset the HF Space env for the current task.
      2. Feed the completion as a single agent action.
      3. Return the composite reward score.
    """
    rewards = []
    task_id = kwargs.get("task_id", "easy")

    for prompt, completion in zip(prompts, completions):
        # Completion is the raw model output — parse action JSON out of it
        action_dict = _extract_json(completion)
        if action_dict is None:
            rewards.append(0.0)  # malformed = worst reward
            continue

        # Reset env
        reset = _call_space("/reset", {"task_id": task_id, "seed": 42})
        if reset is None:
            rewards.append(0.0)
            continue

        # Step
        step = _call_space("/step", {"action": action_dict})
        if step is None:
            rewards.append(0.0)
            continue

        reward_score = step.get("reward", {}).get("score", 0.0)
        rewards.append(float(reward_score))

    return rewards


# %%
# Cell 5 — Build prompts dataset
# We need a collection of "starting observations" that GRPO will generate completions for.
# Easiest: pre-compute 50 Observation snapshots by repeatedly calling /reset with different seeds.

PROMPT_COUNT = 50

def fetch_prompt_obs(seed: int, task: str = "easy") -> str:
    """Reset env with given seed, format the resulting Observation as a prompt string."""
    obs_json = _call_space("/reset", {"task_id": task, "seed": seed})
    if obs_json is None:
        return ""
    # Minimal one-liner prompt — just enough structure for the model to reason over
    campaigns = obs_json.get("active_campaigns", [])
    metrics = obs_json.get("latest_metrics", {})
    smb = obs_json.get("smb_profile", {})
    prompt = (
        f"Business: {smb.get('name')} ({smb.get('industry')})\n"
        f"Budget remaining: ₹{obs_json.get('total_budget_remaining_inr'):.0f}\n"
        f"Active campaigns: {len(campaigns)}\n"
        f"Latest metrics: {json.dumps(metrics)}\n"
        f"What action should the AI Ad Manager take? Respond with JSON only."
    )
    return prompt

print("Fetching prompts from HF Space...")
prompts = []
for s in range(PROMPT_COUNT):
    p = fetch_prompt_obs(seed=s)
    if p:
        prompts.append({"prompt": p, "task_id": "easy"})

print(f"Collected {len(prompts)} prompts")
from datasets import Dataset
prompt_ds = Dataset.from_list(prompts)

# %%
# Cell 6 — GRPO training
from trl import GRPOConfig, GRPOTrainer

cfg = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_generations=NUM_GENERATIONS,          # 2-GRPO
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=LEARNING_RATE,
    max_steps=TOTAL_STEPS,
    logging_steps=5,
    save_steps=50,
    save_strategy="steps",
    bf16=True,
    report_to=["wandb"],
    temperature=0.7,
    max_prompt_length=512,
    max_completion_length=256,
    seed=42,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[env_reward],
    args=cfg,
    train_dataset=prompt_ds,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"\n✓ GRPO training complete. Saved to {OUTPUT_DIR}")

# %%
# Cell 7 — Push final adapter to HF Hub
# model.push_to_hub("Falgunisharma/smb-ad-manager-grpo", private=True)
# tokenizer.push_to_hub("Falgunisharma/smb-ad-manager-grpo", private=True)

# %%
# Cell 8 — Quick eval: how does the trained model perform?
# Uncomment to run a baseline vs trained comparison.
#
# from src.smb_ads.env import Env
# from src.smb_ads.agent import AdManagerAgent
# ... (pull checkpoint, run evaluation, log to wandb)
