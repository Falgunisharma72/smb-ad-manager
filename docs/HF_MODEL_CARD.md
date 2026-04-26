---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
license: mit
language:
- en
tags:
- reinforcement-learning
- grpo
- trl
- peft
- lora
- meta-ads
- openenv
- india
- ad-management
- llm-agents
---

# SMB Ad Manager — Qwen 2.5 1.5B + GRPO LoRA adapter

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Falgunisharma72/smb-ad-manager/blob/main/notebooks/train_smb_ads.ipynb)
[![Live env](https://img.shields.io/badge/HF%20Space-live-blue)](https://falgunisharma-smb-ad-manager.hf.space)
[![Demo site](https://img.shields.io/badge/demo-vercel-black)](https://smb-ad-manager.vercel.app)

> **Submission for the Scaler OpenEnv Hackathon (April 2026).**

A LoRA adapter on top of `Qwen/Qwen2.5-1.5B-Instruct`, RL-refined against a
**reward-hardened, OpenEnv-compliant Meta Ads simulator** that scores every
action on 5 reward components and 5 anti-hack detectors simultaneously.

**+73% reward improvement** over the SFT baseline. Trained on a single
Colab L4 in 51 minutes.

---

## What this model does

Manages Meta Ads campaigns for Indian small businesses. Given a business
brief and current campaign state, it emits a JSON action choosing one of 8
Meta Marketing API tools (`create_campaign`, `pause_ad`, `update_budget`,
`rewrite_creative`, etc.) along with reasoning.

It was trained against a calibrated simulator
([WordStream India](https://www.wordstream.com/blog/ws/2024/03/05/facebook-advertising-benchmarks)
+ [Meta Ad Library](https://www.facebook.com/ads/library/) benchmarks)
where the reward isn't just outcome — it's whether the
agent achieved that outcome *without taking shortcuts*. Five anti-hack
detectors fire alongside the reward signal:

- `mass_pause` — pausing ads to artificially boost ROAS by reducing spend
- `quality_floor` — running ads below the env's quality threshold
- `hallucinated_citation` — citing metrics it never actually fetched
- `action_spam` — repeating the same action to inflate step count
- `policy_ignore` — failing to detect mid-episode policy drift

Every detector firing is logged separately to Weights & Biases.

---

## How to load and run

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(model, "Falgunisharma/smb-ad-manager-grpo")
model.eval()

messages = [
    {"role": "system", "content": "You are an AI Ad Manager. Return a JSON action."},
    {"role": "user", "content": "Business: Priya's Candles (skincare). Budget remaining: INR 8500. ..."},
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True)
print(tok.decode(out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

The full env service (FastAPI on HF Spaces) is at
<https://falgunisharma-smb-ad-manager.hf.space>. The GRPO training script
calls `/reset` and `/step` on this Space for every rollout — so you can
score this model's outputs against the same env it was trained on.

---

## Training pipeline

```
   100 hand-crafted SFT examples            ┌──────────────────┐
              │                              │  Live HF Space   │
              ▼                              │  reward signal   │
  ┌──────────────────────┐                  └────────┬─────────┘
  │  Qwen 2.5 1.5B base  │                           │
  └──────────┬───────────┘                           │
             │                                        │
       Stage 1: SFT                                   │
       LoRA r=16, bf16, 3 epochs                      │
             │                                        │
             ▼                                        │
  ┌──────────────────────┐                            │
  │  SFT-warmed model    │                            │
  └──────────┬───────────┘                            │
             │                                        │
       Stage 2: GRPO  ◄────── reward per rollout ─────┘
       Group size 2 · temperature 0.7
       200 steps · TRL GRPOTrainer
             │
             ▼
  ┌──────────────────────┐
  │  This adapter        │   ← you are here
  └──────────────────────┘
```

### SFT warm-start

100 hand-crafted examples. LoRA r=16, alpha=32, bf16, 4-bit base via
bitsandbytes. 3 epochs.

| Metric | Start | End |
|---|---:|---:|
| Train loss | 2.31 | **0.17** |
| Token accuracy | 57% | **95%** |
| Entropy | — | 0.21 |
| Runtime | — | ~5 min on L4 |

### GRPO RL refinement

The reward function calls our deployed HF Space on every rollout. Group
size 2 (2-GRPO, [arxiv 2510.00977](https://arxiv.org/abs/2510.00977)),
200 steps, learning rate 5e-6, β=0.04.

![GRPO mean reward over 200 steps](https://raw.githubusercontent.com/Falgunisharma72/smb-ad-manager/main/frontend/public/charts/grpo_1_5b_reward.png)

| Metric | Start | End | Δ |
|---|---:|---:|---:|
| Mean reward | 0.41 | **0.71** | **+73%** |
| Reward std | > 0.05 (healthy) | > 0.05 | — |
| Runtime | — | 51 min on L4 | — |

The 5-component breakdown — `r1_roas` lifts to 0.62, `r2/r4/r5` saturate at
1.0 (provably policy-clean, budget-clean, no fabricated metrics), `r3_format`
stays at 0.0 (documented honest weakness):

![5-component reward breakdown - r1 0.62, r2 1.0, r3 0.0, r4 1.0, r5 1.0](https://raw.githubusercontent.com/Falgunisharma72/smb-ad-manager/main/docs/metrics_breakdown3.png)

Public W&B run: <https://wandb.ai/f-banasthali-vidyapith/smb-ad-manager/runs/n3f4majc>

### What this looks like in the live demo

The deployed website surfaces these results on a dedicated metrics page:

![Metrics hero on the live site - 200 GRPO steps, 51 minutes, +73%](https://raw.githubusercontent.com/Falgunisharma72/smb-ad-manager/main/docs/metrics_breakdown1.png)

And lets a small business owner experience the agent in real time:

![Founder Mode form on the live site - business profile + 7-day plan reveal](https://raw.githubusercontent.com/Falgunisharma72/smb-ad-manager/main/docs/founder_demo.png)

For the adversarial side — 5 anti-hack detectors firing on attacker presets:

![Adversarial mode - 5 detectors and 4 attacker presets](https://raw.githubusercontent.com/Falgunisharma72/smb-ad-manager/main/docs/adversial_caught.png)

---

## Reward design — 5 components scored independently

| Component | What it scores |
|---|---|
| `r1_roas_improvement` | Did this step improve ROAS vs yesterday? |
| `r2_policy_compliance` | **Multiplicative kill-switch** — violation → score = 0 |
| `r3_format_compliance` | Did the action match the strict Pydantic schema? |
| `r4_budget_discipline` | Did the agent stay within the monthly budget envelope? |
| `r5_no_cheating` | Did the agent only cite metrics it actually fetched? |

The kill-switch on `r2` is the single most important design choice — it
makes any policy violation a *terminal* event for the reward, regardless
of how good the rest of the action was.

---

## Scaling study — we also tried 3B, and the wall is interesting

We ran the same pipeline on Qwen 2.5 3B. **It did not converge.** Diagnosis
across two iterations and a final 5-minute pre-flight test revealed the
root cause was not GRPO config but **capacity-driven over-generalization
in the SFT-3B model** — the 3B was so confident in its prior over
"ad-manager API" that it hallucinated tool names rather than copying from
the explicit 8-tool list in the system prompt.

![3B v2 reward stuck at 0.35](https://raw.githubusercontent.com/Falgunisharma72/smb-ad-manager/main/frontend/public/charts/grpo_3b_v2_flat.png)

A 5-minute diagnostic histogram (20 sampled completions, all returned 422
from the env, all hallucinated tool names) confirmed it. Full writeup:
[SUBMISSION.md scaling study](https://github.com/Falgunisharma72/smb-ad-manager/blob/main/SUBMISSION.md#scaling-study--qwen-25-3b-did-not-converge-research-finding).

---

## Honest limitations

1. **Trained agent prefers `create_*` tools** over modifying running campaigns. Reliable partial credit but small lift over a noop. *Fix:* train longer or remove `r1` partial credit.
2. **`r3` (format compliance) stays at 0.0** — model emits valid JSON but uses `daily_budget` instead of `daily_budget_inr`. *Fix:* schema-correct SFT examples.
3. **Hallucinated tool names** (`creative_curation`, `creative_selection`) → env returns 422. *Fix:* explicit tool list in every prompt template.

These are surfaced on the live demo's `/metrics` page — judges can verify.

---

## Links

- **Source code:** <https://github.com/Falgunisharma72/smb-ad-manager>
- **Full submission writeup:** <https://github.com/Falgunisharma72/smb-ad-manager/blob/main/SUBMISSION.md>
- **Live env (HF Space):** <https://falgunisharma-smb-ad-manager.hf.space>
- **Live demo (Vercel):** <https://smb-ad-manager.vercel.app> · login `admin` / `hackathon2026`
- **Click-to-run training notebook (in-repo):** [Open in Colab](https://colab.research.google.com/github/Falgunisharma72/smb-ad-manager/blob/main/notebooks/train_smb_ads.ipynb)
- **Live training Colab #1:** <https://colab.research.google.com/drive/1MMqERXe2nzhOcKeIrdWSV2gT0u4V-cXy?usp=sharing>
- **Live training Colab #2:** <https://colab.research.google.com/drive/1ZwiJjy9TJoqx5G54xOKkdlq41VhexJ1S?usp=sharing>
- **Sibling adapters:**
  - SFT 1.5B: <https://huggingface.co/Falgunisharma/smb-ad-manager-sft>
  - SFT 3B: <https://huggingface.co/Falgunisharma/smb-ad-manager-sft-3b>
  - GRPO 3B v2 (research artifact): <https://huggingface.co/Falgunisharma/smb-ad-manager-grpo-3b-v2>

## Citations

- DeepSeek-R1 (GRPO origin) — [arxiv 2501.12948](https://arxiv.org/abs/2501.12948)
- 2-GRPO — [arxiv 2510.00977](https://arxiv.org/abs/2510.00977)
- OpenEnv specification — [github.com/huggingface/openenv](https://github.com/huggingface/openenv)

## Team

**Team name — Sarthak's team**

- Falguni Sharma
- Sarthak Kala
- Shrishty Kothiyal

## License

MIT.
