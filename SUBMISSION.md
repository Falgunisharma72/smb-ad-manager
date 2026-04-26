# SMB Ad Manager — Hackathon Submission

**Team:** Falguni Sharma · Sarthak · Shrishty
**Track:** Theme 3.1 (Professional World Modeling) — *primary*; also hits Themes 1, 2, 4
**Bonus tracks:** Patronus AI · Halluminate · Scaler AI Labs

---

## TL;DR

A reward-hardened, OpenEnv-compliant Meta Ads RL environment for training LLM
agents to manage Indian SMB ad accounts — calibrated against WordStream + Meta
Ad Library, with **5 reward functions + 5 anti-hack detectors** as the
differentiator. We trained Qwen 2.5 1.5B in it and got **+73% reward
improvement**; we tried 3B, hit distribution sharpening collapse, and shipped
the failure as honest research.

---

## 🔗 Live links

| Artifact | URL |
|---|---|
| 🌐 Live env (HF Space) | <https://falgunisharma-smb-ad-manager.hf.space> |
| 🎬 Demo website (Vercel) | <https://smb-ad-manager.vercel.app> · login `admin` / `hackathon2026` |
| 🤖 SFT 1.5B adapter | <https://huggingface.co/Falgunisharma/smb-ad-manager-sft> |
| 🚀 GRPO 1.5B adapter | <https://huggingface.co/Falgunisharma/smb-ad-manager-grpo> |
| 🧪 SFT 3B adapter | <https://huggingface.co/Falgunisharma/smb-ad-manager-sft-3b> |
| 📊 W&B run | <https://wandb.ai/f-banasthali-vidyapith/smb-ad-manager/runs/n3f4majc> |
| 💻 Source | <https://github.com/Falgunisharma72/smb-ad-manager> |

---

## What we built

| Artifact | What it is |
|---|---|
| **Environment** (`src/smb_ads/`) | OpenEnv-compliant FastAPI service · 15 strict Pydantic schemas · 8 mock Meta Marketing API tools · 93 passing tests |
| **Reward stack** (`src/smb_ads/rewards.py`) | 5 reward components + 5 anti-hack detectors, all logged separately |
| **Showcase site** (`frontend/`) | Next.js 15 deployed on Vercel · 6 routes hitting the live HF Space |

---

## Dataset & calibration

| Source | What we used it for |
|---|---|
| **WordStream 2024-2025 Facebook benchmarks** ([link](https://www.wordstream.com/blog/ws/2024/03/05/facebook-advertising-benchmarks)) | Per-vertical baseline CTR / conversion rate / audience pressure |
| **Meta Ads Library** | Reach / impression spot-checks across the 3 verticals |
| **Indian-market CPM rates** (5–8× lower than US) | Per-impression cost realism — the reason ROAS in our scenarios is meaningful |
| **Hand-crafted SFT data** (`training/sft_data.jsonl`) | 100 examples, 9-tool action grammar, every example schema-valid |

### Verticals we calibrated (3 industries × benchmarks)

| Industry | Baseline CTR | Conv rate | AOV | CPM (INR) | Pressure |
|---|---:|---:|---:|---:|---:|
| Skincare / D2C beauty | 1.4% | 2.5% | ₹850 | ₹95 | 1.1× |
| Food delivery | 1.0% | 3.5% | ₹420 | ₹75 | 1.3× |
| Fitness apps | 1.6% | 1.8% | ₹250 | ₹85 | 1.0× |

### SMB scenarios

**10 realistic SMB profiles** spanning ₹5K – ₹50K/month budgets across the 3
verticals — Priya's Handmade Candles, Glow Naturally Skincare, Thali Express,
FitTrack India, and 6 more. Each scenario seeds reproducible 3-7 day episodes.

---

## Training pipeline

```
   100 hand-crafted SFT examples            ┌──────────────────┐
              │                              │  Live HF Space   │
              ▼                              │  reward signal   │
  ┌──────────────────────┐                  └────────┬─────────┘
  │  Qwen 2.5 base       │                           │
  │  (1.5B or 3B)        │                           │
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
       Group size 2 (1.5B) / 4 (3B-v2)
       200 steps · TRL GRPOTrainer
             │
             ▼
  ┌──────────────────────┐
  │  GRPO-refined agent  │
  └──────────────────────┘
```

Both stages run on **a single Colab L4 in under 90 minutes** wall-clock.

---

## Results

### Stage 1 — SFT warm-start

| Model | Loss start → end | Token accuracy | Entropy end | Runtime |
|---|---:|---:|---:|---:|
| Qwen 2.5 1.5B | 2.31 → **0.17** | 95.0% | 0.21 | ~5 min on L4 |
| Qwen 2.5 3B   | 2.32 → **0.165** | 95.6% | 0.18 | ~25 min on L4 |

Both converged. The 3B's slightly lower entropy (0.18) becomes load-bearing in
Stage 2 — see **Distribution sharpening** below.

### Stage 2 — GRPO refinement

| Model | Reward start | Reward end | Δ | grad_norm | reward_std | Result |
|---|---:|---:|---:|---:|---:|---|
| **1.5B + GRPO** | 0.41 | **0.71** | **+73%** | healthy | > 0.05 | ✅ converged cleanly |
| 3B + GRPO (v1, default config) | 0.35 | 0.35 | +0% | 0 | 0 | ⚠ distribution sharpening collapse |
| 3B + GRPO (v2, anti-collapse) | _running_ | _running_ | _running_ | — | — | _result will be appended_ |

### What "baseline" means in this comparison

The **baseline is the SFT-only model** — i.e., the model after Stage 1, before
any RL. Its reward score on the env is 0.41 for 1.5B (and 0.35 for 3B). The
GRPO row shows lift over that baseline. The +73% number is therefore
**SFT-only → SFT+GRPO**, on the same env, same 50 prompts, same seed.

### 1.5B GRPO — reward variance stayed healthy across 200 steps

![Reward std over 200 GRPO steps](frontend/public/charts/1.png)

This is the *opposite* of the 3B v1 collapse: every batch has non-zero reward
variance, so group-relative advantage actually carries signal. Spikes around
step 50 are the model exploring new tools, plateaus are it consolidating gains.

> **For richer visuals** — screenshots of the live `/founder`, `/adversarial`,
> and `/metrics` pages will be added to `docs/screenshots/` once captured.

---

## Distribution sharpening collapse — research finding (3B v1)

The 3B model's SFT converged to a near-deterministic policy. At the GRPO
sampling temperature of 0.7, **every rollout in a group of 2 was identical**.
This makes the group-relative advantage `(reward − mean) / std` exactly zero,
every step — so no gradient, no learning.

**Diagnostic signature:**

- `frac_reward_zero_std = 1.0` (every batch had zero variance)
- `grad_norm = 0` (no gradient signal)
- `loss = 0` (pure no-op)

**v2 anti-collapse config** (currently training):

| Knob | v1 | v2 | Why |
|---|---:|---:|---|
| Temperature | 0.7 | **1.0** | Wider sampling distribution |
| Group size | 2 | **4** | More rollouts → more chance of variance |
| KL coefficient `β` | 0.04 | **0.0** | Stop pulling completions back to the SFT mode |
| Learning rate | 5e-6 | **1e-5** | Bigger step when signal does appear |
| Top-p | 1.0 | **0.95** | Tail-cut to keep coherence at higher temp |

---

## Honest limitations

We surface these on the live `/metrics` page too — judges can verify.

1. **Trained agent prefers `create_*` tools** over modifying running campaigns. Reliable partial credit but small lift over a noop. *Fix:* train longer or remove `r1` partial credit.
2. **`r3` (format compliance) stays at 0.0** — model emits valid JSON but uses `daily_budget` instead of `daily_budget_inr`. *Fix:* dict-shape match or schema-correct SFT examples.
3. **Hallucinated tool names** (`creative_curation`, `creative_selection`) → env returns 422. *Fix:* explicit tool list in every prompt template.
4. **3B distribution sharpening collapse** (above) — documented as research finding rather than hidden.

---

## Themes & bonus tracks

| Theme | How we hit it |
|---|---|
| **3.1 Professional World Modeling** *(primary)* | Calibrated Meta Ads ecosystem, user response, ad auction, policy enforcement |
| **1 Multi-Agent** | 3 simulated actors (user model · ad auction · policy enforcer) act concurrently against the agent |
| **2 Long-Horizon** | 3-7 day episodes with mid-week policy drift |
| **4 Self-Improvement** | GRPO on a live reward signal |

| Bonus track | Hook |
|---|---|
| 🎯 **Patronus AI** | Mid-episode `p6_health_disclaimer` drift — agent detects via `get_policy_updates`, repairs via `rewrite_creative` |
| 🎯 **Halluminate** | `r5_no_cheating` verifies the agent only cites metrics it actually fetched |
| 🎯 **Scaler AI Labs** | 5 reward components + 5 anti-hack detectors logged separately, visible per step |

---

## Tech stack

Python 3.11 · FastAPI · Pydantic 2 · pytest · HuggingFace Spaces (Docker) ·
Vercel · Qwen 2.5 (1.5B / 3B Instruct) · HF TRL · peft (LoRA) · bitsandbytes
4-bit · Colab Pro L4 · Weights & Biases · Next.js 15 · Tailwind · Framer Motion.

---

## Citations

- DeepSeek-R1 (GRPO origin) — [arxiv 2501.12948](https://arxiv.org/abs/2501.12948)
- 2-GRPO (12.5% rollouts of standard GRPO) — [arxiv 2510.00977](https://arxiv.org/abs/2510.00977)
- OpenEnv specification — [github.com/huggingface/openenv](https://github.com/huggingface/openenv)

---

*MIT licensed. Fork freely.*
