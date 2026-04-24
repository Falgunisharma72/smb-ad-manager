"""GRPO training — vanilla TRL + peft, NO Unsloth.

Bypasses Unsloth's fast_lora kernel (which has a dtype bug in 2026.4.8).
Uses standard HuggingFace transformers + peft + TRL stack — slower than
Unsloth but actually works.

Runtime on L4: ~2.5-3 hours for 200 steps. Still fits in Colab Pro.
Output: ./checkpoints/grpo_final/
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Monkeypatch — transformers 5.x removed TRANSFORMERS_CACHE but llm_blender (used
# transitively by TRL's judges module) still imports it. Provide a shim.
import transformers.utils.hub as _hub
if not hasattr(_hub, "TRANSFORMERS_CACHE"):
    try:
        from huggingface_hub import constants as _hf_const
        _hub.TRANSFORMERS_CACHE = _hf_const.HF_HUB_CACHE
    except Exception:
        _hub.TRANSFORMERS_CACHE = str(Path.home() / ".cache" / "huggingface" / "hub")

# ─── Config ──────────────────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SFT_ADAPTER_PATH = "./checkpoints/sft_adapter"
OUTPUT_DIR = "./checkpoints/grpo_final"
SPACE_URL = os.environ.get("SPACE_URL", "https://falgunisharma-smb-ad-manager.hf.space")
MAX_SEQ_LENGTH = 1024
NUM_GENERATIONS = 2
TOTAL_STEPS = 200
LEARNING_RATE = 5e-6
WANDB_PROJECT = "smb-ad-manager"
PROMPT_COUNT = 50

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
os.environ["WANDB_PROJECT"] = WANDB_PROJECT
print(f"Space URL: {SPACE_URL}")
print(f"Training: {TOTAL_STEPS} steps, group size {NUM_GENERATIONS} (2-GRPO)")

# ─── Load model (vanilla transformers, no Unsloth) ──────────────────────
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

# Load SFT adapter and merge it into base so GRPO starts from SFT knowledge
print(f"Loading SFT adapter from {SFT_ADAPTER_PATH} + merging...")
sft_model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
merged_model = sft_model.merge_and_unload()
# merged_model is the SFT-refined base (no LoRA layers anymore)

# Attach a FRESH LoRA adapter for GRPO to train
print("Attaching fresh LoRA for GRPO...")
grpo_lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ─── Reward function (with partial credit, SSL-tolerant) ────────────────
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
    """TRL-compatible reward function. One reward per completion.

    Partial credit:
      no JSON        → 0.00
      not a dict     → 0.05
      has tool       → +0.10
      has reasoning  → +0.05
      has args       → +0.05
      valid env call → max(base+0.15, env_score)
    """
    rewards = []
    for completion in completions:
        # Extract text (TRL can pass various formats)
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

# ─── TRL GRPO training ──────────────────────────────────────────────────
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
    temperature=0.7,
    max_prompt_length=512,
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

print("\n=== Starting GRPO training ===\n")
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n✓ GRPO training complete. Saved to {OUTPUT_DIR}")
