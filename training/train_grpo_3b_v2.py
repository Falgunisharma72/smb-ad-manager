"""GRPO training for Qwen 2.5 3B — v2, fixes distribution sharpening collapse.

v1 result: reward stuck at 0.35, grad_norm=0, loss=0, frac_reward_zero_std=1.0.
Cause: SFT-warmed 3B sampled near-deterministically at temp=0.7. Every rollout
in a group of 2 was identical → group-relative advantage was always 0 → no
gradient signal.

v2 fixes (all aimed at restoring rollout variance):
  - temperature 0.7 → 1.0          (wider sampling distribution)
  - num_generations 2 → 4          (more rollouts per prompt = more chance of variance)
  - learning_rate 5e-6 → 1e-5      (bigger step when signal does appear)
  - beta 0.04 → 0.0                (kill KL penalty pulling completions back to SFT mode)
  - top_p 1.0 → 0.95               (tail-cut to keep coherence at higher temp)

Runtime on L4: ~70 min for 200 steps (group=4 doubles rollout cost).
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import transformers.utils.hub as _hub
if not hasattr(_hub, "TRANSFORMERS_CACHE"):
    try:
        from huggingface_hub import constants as _hf_const
        _hub.TRANSFORMERS_CACHE = _hf_const.HF_HUB_CACHE
    except Exception:
        _hub.TRANSFORMERS_CACHE = str(Path.home() / ".cache" / "huggingface" / "hub")

# ─── Config ──────────────────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
SFT_ADAPTER_PATH = "./checkpoints/sft_adapter_3b"
OUTPUT_DIR = "./checkpoints/grpo_final_3b_v2"
SPACE_URL = os.environ.get("SPACE_URL", "https://falgunisharma-smb-ad-manager.hf.space")
NUM_GENERATIONS = 4          # was 2 — more rollouts per prompt
TOTAL_STEPS = 200
LEARNING_RATE = 1e-5         # was 5e-6
TEMPERATURE = 1.0            # was 0.7
TOP_P = 0.95                 # was 1.0 (default)
BETA = 0.0                   # was 0.04 — disable KL penalty
WANDB_PROJECT = "smb-ad-manager"
PROMPT_COUNT = 50

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
os.environ["WANDB_PROJECT"] = WANDB_PROJECT
os.environ["WANDB_NAME"] = "grpo-3b-v2-anti-collapse"
print(f"Space URL: {SPACE_URL}")
print(f"Training: {TOTAL_STEPS} steps, group={NUM_GENERATIONS}, temp={TEMPERATURE}, beta={BETA}")

# ─── Load model ──────────────────────────────────────────────────────────
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig

print("Loading base model (3B)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

print(f"Loading SFT adapter from {SFT_ADAPTER_PATH} + merging...")
sft_model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
merged_model = sft_model.merge_and_unload()

print("Attaching fresh LoRA for GRPO...")
grpo_lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ─── Reward function ─────────────────────────────────────────────────────
import requests
import urllib3
urllib3.disable_warnings()

SESSION = requests.Session()
SESSION.verify = False
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_action(text):
    m = _JSON_RE.search(text.strip())
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _call_space(path, payload, retries=2):
    for attempt in range(retries):
        try:
            r = SESSION.post(f"{SPACE_URL}{path}", json=payload, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(1)
    return None


def env_reward(completions, prompts=None, **kwargs):
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            text = completion[-1].get("content", "") if completion else ""
        elif isinstance(completion, dict):
            text = completion.get("content", "")
        else:
            text = str(completion)

        action = _extract_action(text)
        if action is None:
            rewards.append(0.0)
            continue
        if not isinstance(action, dict):
            rewards.append(0.05)
            continue

        has_tool = "tool" in action
        has_reasoning = "reasoning" in action
        has_args = "args" in action
        base = 0.05
        base += 0.10 if has_tool else 0
        base += 0.05 if has_reasoning else 0
        base += 0.05 if has_args else 0

        if not has_tool:
            rewards.append(base)
            continue

        r = _call_space("/reset", {"task_id": "easy", "seed": 42})
        if r is None:
            rewards.append(base)
            continue
        s = _call_space("/step", {"action": action})
        if s is None:
            rewards.append(base + 0.10)
            continue

        env_score = float(s.get("reward", {}).get("score", 0.0))
        rewards.append(max(base + 0.15, env_score))
    return rewards


# ─── Build prompts dataset ──────────────────────────────────────────────
print(f"Fetching {PROMPT_COUNT} prompts from HF Space...")
prompts_data = []
for s in range(PROMPT_COUNT):
    obs = _call_space("/reset", {"task_id": "easy", "seed": s})
    if obs is None:
        continue
    smb = obs.get("smb_profile", {})
    user_msg = (
        f"Business: {smb.get('name')} ({smb.get('industry')})\n"
        f"Budget remaining: INR {obs.get('total_budget_remaining_inr', 0):.0f}\n"
        f"Active campaigns: {len(obs.get('active_campaigns', []))}\n"
        f"Latest metrics: {json.dumps(obs.get('latest_metrics', {}))}\n"
        "Return a JSON action like {\"tool\": \"...\", \"args\": {...}, \"reasoning\": \"...\"}"
    )
    prompts_data.append({
        "prompt": [
            {"role": "system", "content": "You are an AI Ad Manager. Return a JSON action."},
            {"role": "user", "content": user_msg},
        ],
    })

if not prompts_data:
    raise RuntimeError("Could not fetch prompts. Check SPACE_URL.")
print(f"Collected {len(prompts_data)} prompts")

from datasets import Dataset
prompt_ds = Dataset.from_list(prompts_data)

if not hasattr(merged_model, "warnings_issued"):
    merged_model.warnings_issued = {}

# ─── TRL GRPO training (anti-collapse config) ────────────────────────────
from trl import GRPOConfig, GRPOTrainer

cfg = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_generations=NUM_GENERATIONS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=LEARNING_RATE,
    max_steps=TOTAL_STEPS,
    logging_steps=5,
    save_steps=50,
    bf16=True,
    report_to=["wandb"],
    temperature=TEMPERATURE,
    top_p=TOP_P,
    beta=BETA,
    max_completion_length=256,
    seed=42,
    remove_unused_columns=False,
)

trainer = GRPOTrainer(
    model=merged_model,
    processing_class=tokenizer,
    reward_funcs=[env_reward],
    args=cfg,
    train_dataset=prompt_ds,
    peft_config=grpo_lora_config,
)

print("\n=== Starting GRPO 3B v2 (anti-collapse) ===\n")
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n✓ GRPO 3B v2 training complete. Saved to {OUTPUT_DIR}")
