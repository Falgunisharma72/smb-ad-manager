"""Custom minimal GRPO — DGX-compatible variant.

REINFORCE with group-relative baseline (the core idea of GRPO, implemented
by hand). No TRL. No bitsandbytes. No wandb.

Works on:
  - nvcr.io/nvidia/pytorch:22.12-py3
  - Driver 470 via CUDA 11.8 forward-compat
  - Python 3.8, NVIDIA torch 1.14.0a0
  - Any fp16-capable GPU (A100 80GB more than enough for Qwen 1.5B)

Algorithm:
  For each step:
    1. Sample BATCH_SIZE prompts
    2. For each prompt, generate NUM_GENERATIONS completions (sampling, temp > 0)
    3. Score each completion via HTTP to the HF Space env (/reset + /step)
    4. Compute group-relative advantages: (r - group_mean) / (group_std + eps)
    5. Re-run forward to get log-probs of generated tokens (with gradients)
    6. Loss = -(advantage * mean_log_prob).mean()
    7. Backprop + optimizer step

Logs reward + loss to TensorBoard at ./tb_logs/grpo/.

Expected runtime on A100: ~20-40 min for 200 steps (group size 2, batch 2).
Output: ./checkpoints/grpo_final/
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ─── Config ──────────────────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SFT_ADAPTER_PATH = "./checkpoints/sft_adapter"
OUTPUT_DIR = "./checkpoints/grpo_final"
TB_LOG_DIR = "./tb_logs/grpo"
SPACE_URL = os.environ.get("SPACE_URL", "https://falgunisharma-smb-ad-manager.hf.space")

TOTAL_STEPS = 200
NUM_GENERATIONS = 2          # 2-GRPO (arxiv 2510.00977 — compute-efficient)
BATCH_SIZE = 2               # prompts per step
MAX_NEW_TOKENS = 256
LR = 1e-5
TEMPERATURE = 0.7
TOP_P = 0.9
PROMPTS_PER_EPISODE = 50     # size of the prompt pool we sample from
LOG_EVERY = 5
SAVE_EVERY = 50

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(TB_LOG_DIR).mkdir(parents=True, exist_ok=True)

# ─── Imports ─────────────────────────────────────────────────────────────
import requests
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from torch.utils.tensorboard import SummaryWriter

print(f"Space URL: {SPACE_URL}")
print(f"Training: {TOTAL_STEPS} steps, group={NUM_GENERATIONS}, batch={BATCH_SIZE}")
print(f"CUDA: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0)}")

# ─── Load model (base + SFT adapter) ─────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(SFT_ADAPTER_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map={"": 0},
)
model = PeftModel.from_pretrained(base, SFT_ADAPTER_PATH, is_trainable=True)
model.print_trainable_parameters()
model.train()

# ─── Reward function (HTTP into the HF Space env) ────────────────────────
SESSION = requests.Session()
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_action(text: str):
    m = _JSON_RE.search(text.strip())
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _call_space(path: str, payload: dict, retries: int = 3):
    for attempt in range(retries):
        try:
            r = SESSION.post(f"{SPACE_URL}{path}", json=payload, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                print(f"[space] final attempt failed: {e}")
            time.sleep(2 ** attempt)
    return None


def env_reward(completion_text: str, task_id: str = "easy") -> float:
    action = _extract_action(completion_text)
    if action is None:
        return 0.0
    r = _call_space("/reset", {"task_id": task_id, "seed": 42})
    if r is None:
        return 0.0
    s = _call_space("/step", {"action": action})
    if s is None:
        return 0.0
    return float(s.get("reward", {}).get("score", 0.0))


# ─── Build prompt pool from the env ──────────────────────────────────────
print("\nFetching prompts from HF Space...")
prompts_data = []
for s in range(PROMPTS_PER_EPISODE):
    obs = _call_space("/reset", {"task_id": "easy", "seed": s})
    if obs is None:
        continue
    smb = obs.get("smb_profile", {})
    user_msg = (
        f"Business: {smb.get('name')} ({smb.get('industry')})\n"
        f"Budget remaining: INR {obs.get('total_budget_remaining_inr', 0):.0f}\n"
        f"Active campaigns: {len(obs.get('active_campaigns', []))}\n"
        f"Latest metrics: {json.dumps(obs.get('latest_metrics', {}))}\n"
        "Return a JSON action."
    )
    messages = [
        {"role": "system", "content": "You are an AI Ad Manager. Return a JSON action."},
        {"role": "user", "content": user_msg},
    ]
    prompts_data.append(tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    ))

if not prompts_data:
    raise RuntimeError("Could not fetch any prompts from the HF Space. Check SPACE_URL.")
print(f"Collected {len(prompts_data)} prompts")

# ─── Optimizer + TensorBoard ─────────────────────────────────────────────
trainable_params = [p for p in model.parameters() if p.requires_grad]
print(f"Optimizing {sum(p.numel() for p in trainable_params) / 1e6:.2f}M LoRA params")
optimizer = torch.optim.AdamW(trainable_params, lr=LR)
writer = SummaryWriter(log_dir=TB_LOG_DIR)

# ─── Training loop ───────────────────────────────────────────────────────
print(f"\n=== Starting training: {TOTAL_STEPS} steps ===\n")

for step in range(TOTAL_STEPS):
    step_rewards = []
    step_losses = []
    batch_prompts = random.sample(prompts_data, min(BATCH_SIZE, len(prompts_data)))

    optimizer.zero_grad()

    for prompt_text in batch_prompts:
        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to("cuda")
        prompt_len = prompt_ids.shape[1]

        # Generate NUM_GENERATIONS completions (no gradient during generate)
        with torch.no_grad():
            gens = model.generate(
                prompt_ids.repeat(NUM_GENERATIONS, 1),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion_ids = gens[:, prompt_len:]  # [G, gen_len]

        # Score each completion via the env
        rewards = []
        for i in range(NUM_GENERATIONS):
            txt = tokenizer.decode(completion_ids[i], skip_special_tokens=True)
            rewards.append(env_reward(txt))
        step_rewards.extend(rewards)

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device="cuda")

        # Group-relative advantage
        mean = rewards_t.mean()
        std = rewards_t.std(unbiased=False)
        advantages = (rewards_t - mean) / (std + 1e-8)

        # Skip this prompt if all rewards equal (advantage = 0, no signal)
        if advantages.abs().sum() < 1e-6:
            continue

        # Re-run forward with gradients to compute log-probs of sampled tokens
        model_out = model(gens)
        logits = model_out.logits  # [G, total_len, vocab]
        # Shift so logits[:, t] predicts token at position t+1
        shift_logits = logits[:, prompt_len - 1 : -1, :]  # [G, gen_len, vocab]
        shift_labels = completion_ids                     # [G, gen_len]

        # Mask out padding tokens in labels
        mask = (shift_labels != tokenizer.pad_token_id).float()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            2, shift_labels.unsqueeze(-1).clamp(min=0)
        ).squeeze(-1)
        token_log_probs = token_log_probs * mask

        # Mean log-prob per sequence
        seq_len = mask.sum(dim=1).clamp(min=1)
        mean_log_probs = token_log_probs.sum(dim=1) / seq_len

        # REINFORCE with group baseline
        loss = -(advantages * mean_log_probs).mean()
        # Normalize by accumulation (we have BATCH_SIZE prompts per step)
        loss = loss / len(batch_prompts)
        loss.backward()
        step_losses.append(loss.item() * len(batch_prompts))

    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
    optimizer.step()

    # ─── Logging ────────────────────────────────────────────────────────
    if step % LOG_EVERY == 0 or step == TOTAL_STEPS - 1:
        mean_reward = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0
        mean_loss = sum(step_losses) / len(step_losses) if step_losses else 0.0
        best_reward = max(step_rewards) if step_rewards else 0.0
        print(f"[step {step:4d}] reward_mean={mean_reward:.3f} "
              f"reward_best={best_reward:.3f} loss={mean_loss:.4f}")
        writer.add_scalar("reward/mean", mean_reward, step)
        writer.add_scalar("reward/best", best_reward, step)
        writer.add_scalar("loss/policy", mean_loss, step)

    # ─── Periodic save ─────────────────────────────────────────────────
    if (step + 1) % SAVE_EVERY == 0:
        ck = f"{OUTPUT_DIR}/step_{step+1:04d}"
        model.save_pretrained(ck)
        print(f"  [checkpoint] saved to {ck}")

# ─── Final save ──────────────────────────────────────────────────────────
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
writer.close()
print(f"\n✓ GRPO training complete. Final adapter saved to {OUTPUT_DIR}")
print(f"TensorBoard logs at {TB_LOG_DIR}")
