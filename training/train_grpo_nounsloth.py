# %% [markdown]
# # GRPO Training (non-Unsloth variant) — for CUDA 11.x / driver 470
#
# Starts from the SFT warm-start adapter, runs 2-GRPO against the live HF Space env.
#
# Expected runtime on A100 80GB: ~20-40 min for 200 steps.
#
# Output: `./checkpoints/grpo_final/`
#
# Run: python training/train_grpo_nounsloth.py

# %%
# Cell 1 — Config
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SFT_CHECKPOINT = "./checkpoints/sft_adapter"
OUTPUT_DIR = "./checkpoints/grpo_final"
SPACE_URL = os.environ.get("SPACE_URL", "https://falgunisharma-smb-ad-manager.hf.space")
MAX_SEQ_LENGTH = 2048
NUM_GENERATIONS = 2          # 2-GRPO per arxiv 2510.00977
TOTAL_STEPS = 200
LEARNING_RATE = 1e-5
WANDB_PROJECT = "smb-ad-manager"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
os.environ["WANDB_PROJECT"] = WANDB_PROJECT
print(f"Space URL: {SPACE_URL}")
print(f"Training: {TOTAL_STEPS} steps, group size {NUM_GENERATIONS} (2-GRPO)")

# %%
# Cell 2 — Load SFT-warm-started model (base + LoRA adapter from SFT)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(SFT_CHECKPOINT)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.float16,
)
base_model = prepare_model_for_kbit_training(base_model)

# Load the SFT adapter on top
model = PeftModel.from_pretrained(base_model, SFT_CHECKPOINT, is_trainable=True)
print(f"✓ Loaded SFT adapter from {SFT_CHECKPOINT}")
model.print_trainable_parameters()

# %%
# Cell 3 — Reward function: calls the live HF Space env
import json
import re
import time
import requests

_SESSION = requests.Session()
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


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


def _extract_action(text: str) -> dict | None:
    m = _JSON_OBJ_RE.search(text.strip())
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def env_reward(prompts, completions, **kwargs) -> list[float]:
    """TRL-compatible reward: replay each completion once in the env, return score."""
    rewards = []
    task_id = kwargs.get("task_id", "easy")
    for _prompt, completion in zip(prompts, completions):
        action = _extract_action(completion)
        if action is None:
            rewards.append(0.0)
            continue
        reset = _call_space("/reset", {"task_id": task_id, "seed": 42})
        if reset is None:
            rewards.append(0.0)
            continue
        step = _call_space("/step", {"action": action})
        if step is None:
            rewards.append(0.0)
            continue
        rewards.append(float(step.get("reward", {}).get("score", 0.0)))
    return rewards


# %%
# Cell 4 — Build prompts dataset from the live env
PROMPT_COUNT = 50


def fetch_prompt_obs(seed: int, task: str = "easy") -> str | None:
    obs = _call_space("/reset", {"task_id": task, "seed": seed})
    if obs is None:
        return None
    campaigns = obs.get("active_campaigns", [])
    metrics = obs.get("latest_metrics", {})
    smb = obs.get("smb_profile", {})
    return (
        f"Business: {smb.get('name')} ({smb.get('industry')})\n"
        f"Budget remaining: ₹{obs.get('total_budget_remaining_inr'):.0f}\n"
        f"Active campaigns: {len(campaigns)}\n"
        f"Latest metrics: {json.dumps(metrics)}\n"
        f"What action should the AI Ad Manager take? Respond with JSON only."
    )


print(f"Fetching {PROMPT_COUNT} prompts from HF Space...")
prompts = []
for s in range(PROMPT_COUNT):
    p = fetch_prompt_obs(seed=s)
    if p:
        prompts.append({"prompt": p, "task_id": "easy"})
print(f"Collected {len(prompts)} prompts")

from datasets import Dataset
prompt_ds = Dataset.from_list(prompts)

# %%
# Cell 5 — GRPO training
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
    save_strategy="steps",
    fp16=True,
    report_to=["tensorboard"],    # use tensorboard; swap to ["wandb"] if you have working wandb
    logging_dir="./tb_logs/grpo",
    temperature=0.7,
    max_prompt_length=512,
    max_completion_length=256,
    seed=42,
    save_safetensors=True,
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
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n✓ GRPO training complete. Saved to {OUTPUT_DIR}")

# %%
# Cell 6 — Optional: push to HF Hub (uncomment to use)
# model.push_to_hub("YOUR-HF-USERNAME/smb-ad-manager-grpo", private=True)
# tokenizer.push_to_hub("YOUR-HF-USERNAME/smb-ad-manager-grpo", private=True)
