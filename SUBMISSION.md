# SMB Ad Manager — Scaler OpenEnv Hackathon Submission

**Team:** Falguni Sharma · Sarthak · Shrishty
**Date:** April 26, 2026
**Track:** Theme 3.1 (Professional World Modeling) — *primary*; Themes 1, 2, 4 also hit
**Bonus tracks claimed:** Patronus AI · Halluminate · Scaler AI Labs

---

## TL;DR

We built a **reward-hardened, OpenEnv-compliant Meta Ads RL environment** for
training LLM agents to manage ad campaigns for Indian SMBs — calibrated against
real-world data, with **5 reward functions + 5 always-on anti-hack detectors**
as the differentiator. Then we trained an agent in it (Qwen 2.5 1.5B + GRPO),
got a clean **+73% reward improvement**, attempted to scale to 3B, **discovered
distribution sharpening collapse**, and shipped both wins and the failure as
honest research.

---

## 🔗 Live links

| Artifact | Link |
|---|---|
| 🌐 Live env service | <https://falgunisharma-smb-ad-manager.hf.space> |
| 🎬 Demo website | <https://smb-ad-manager.vercel.app> *(login `admin` / `hackathon2026`)* |
| 🤖 SFT 1.5B adapter | <https://huggingface.co/Falgunisharma/smb-ad-manager-sft> |
| 🚀 GRPO 1.5B adapter | <https://huggingface.co/Falgunisharma/smb-ad-manager-grpo> |
| 🧪 SFT 3B adapter | <https://huggingface.co/Falgunisharma/smb-ad-manager-sft-3b> |
| 📊 W&B run | <https://wandb.ai/f-banasthali-vidyapith/smb-ad-manager/runs/n3f4majc> |
| 💻 Source | <https://github.com/Falgunisharma72/smb-ad-manager> |

---

## The problem

Indian SMBs spend **₹50,000+/month** on Meta Ads but cannot afford a real growth
marketer. AI agents *could* fill the gap — but no one had built a safe place
where an LLM agent could learn ad-management without burning real budget. We
built that place. Then we trained an agent in it.

---

## What we built — three artifacts

### 1. The environment (`src/smb_ads/`)

OpenEnv-compliant FastAPI service. **Lives at
[falgunisharma-smb-ad-manager.hf.space](https://falgunisharma-smb-ad-manager.hf.space).**

- 15 strict Pydantic 2 schemas (`extra="forbid"`)
- Calibrated user response model (WordStream India + Meta Ad Library)
- 8 mock Meta Marketing API tools (create_campaign, pause_ad, get_metrics, …)
- 3 industries × 10 SMB profiles (skincare, food delivery, fitness apps)
- 5 always-on policies + 1 mid-episode drift event
- 93 passing tests across 6 test files

### 2. The reward stack (`src/smb_ads/rewards.py`)

| Component | What it scores |
|---|---|
| `r1_roas_improvement` | Did this step improve ROAS vs yesterday? |
| `r2_policy_compliance` | **Multiplicative kill-switch** — violation → score = 0 |
| `r3_format_compliance` | Did the action match the strict Pydantic schema? |
| `r4_budget_discipline` | Did the agent stay within the monthly budget envelope? |
| `r5_no_cheating` | Did the agent only cite metrics it actually fetched? |

Plus **5 anti-hack detectors** that fire alongside: `mass_pause`, `quality_floor`,
`hallucinated_citation`, `action_spam`, `policy_ignore`. Logged separately to
W&B and visible to the demo UI in real time.

### 3. The showcase website (`frontend/`)

Next.js 15 + Tailwind + Framer Motion deployed on Vercel. 6 routes — each one
hits the live HF Space:

- `/dashboard` — live env health pill + site map
- `/founder` — flagship demo, fill business brief, watch trained-model 7-day plan reveal
- `/playground` — judge plays as the agent, sees rewards + hacks fire
- `/adversarial` — 4 attacker presets, 5 detectors light up red
- `/metrics` — W&B charts, 5-component reward breakdown, honest limitations
- `/about` — team + architecture + citations

---

## Training pipeline

```
Qwen 2.5 (1.5B or 3B Instruct)
        │
        │ Stage 1: SFT warm-start
        │   • 100 hand-crafted examples
        │   • LoRA r=16, alpha=32, dropout=0.05
        │   • bf16, 4-bit bnb, 3 epochs
        │   • TRL SFTTrainer
        ▼
SFT-warmed model
        │
        │ Stage 2: GRPO RL refinement
        │   • Live HF Space as reward source
        │   • LoRA r=16, alpha=32 (fresh adapter)
        │   • 200 steps, group size 2 (1.5B) or 4 (3B-v2)
        │   • TRL GRPOTrainer
        ▼
GRPO-refined model
```

Both stages run on a single Colab L4 in **under 90 minutes wall-clock**.

---

## Results — comparison

### Stage 1: SFT warm-start

| Model | Loss start | Loss end | Token accuracy | Entropy end | Runtime |
|---|---:|---:|---:|---:|---:|
| Qwen 2.5 1.5B + SFT LoRA | 2.31 | **0.17** | 95.0% | 0.21 | ~5 min on L4 |
| Qwen 2.5 3B + SFT LoRA   | 2.32 | **0.165** | 95.6% | 0.18 | ~25 min on L4 |

Both SFT runs converged cleanly. The 3B's lower entropy (0.18 vs 0.21) becomes
load-bearing in Stage 2 — see below.

### Stage 2: GRPO refinement

| Model | Reward start | Reward end | Δ | grad_norm | reward_std | Result |
|---|---:|---:|---:|---:|---:|---|
| Qwen 2.5 1.5B + GRPO | 0.41 | **0.71** | **+73%** | healthy | > 0.05 | ✅ converged cleanly |
| Qwen 2.5 3B + GRPO (v1) | 0.35 | 0.35 | +0% | 0 | 0 | ⚠ distribution sharpening collapse |
| Qwen 2.5 3B + GRPO (v2) | _running_ | _running_ | _running_ | _running_ | _running_ | _v2 result will be added once training completes_ |

### Why 3B v1 collapsed (research finding)

The SFT-warmed 3B model converged to a near-deterministic policy at the GRPO
sampling temperature of 0.7. Within each group of 2 rollouts, completions were
**identical** — making the group-relative advantage `(reward − mean) / std`
exactly zero, every step. Hence:

- `frac_reward_zero_std = 1.0` (every batch had zero variance)
- `grad_norm = 0` (no gradient signal)
- `loss = 0` (pure no-op)

This is a textbook **distribution sharpening collapse**. The fix tried in v2:

| Config | v1 | v2 (anti-collapse) |
|---|---|---|
| Sampling temperature | 0.7 | **1.0** |
| Group size | 2 | **4** |
| KL coefficient (`beta`) | 0.04 | **0.0** |
| Top-p | 1.0 | **0.95** |
| Learning rate | 5e-6 | **1e-5** |

The combination is designed to widen rollout variance and stop the KL penalty
from pulling completions back toward the SFT mode. v2 result will be appended
when the run completes.

---

## Honest limitations (we publish the bad numbers too)

We deliberately surface what's broken — judges trust honesty over polish.

1. **Trained agent converges to "spam create" policy.** After 200 GRPO steps,
   the 1.5B model heavily prefers `create_campaign` / `create_ad` over actually
   modifying running campaigns. Reliable partial credit, but lift over a noop
   baseline is small. *Fix:* train longer or remove partial credit from `r1`.

2. **Format compliance reward (`r3`) stays at 0.0.** The model outputs valid
   JSON but uses field names like `daily_budget` instead of the schema's
   `daily_budget_inr`. *Fix:* soften `r3` to dict-shape match, or add
   schema-correct examples to SFT data.

3. **Hallucinated tool names.** The 1.5B model occasionally invents tool names
   (`creative_curation`, `creative_selection`); env returns 422. *Fix:* include
   explicit tool list in every prompt template.

4. **3B distribution sharpening collapse** (above). Documented as research
   finding rather than hidden.

These are all visible on the `/metrics` page of the live demo with proposed
fixes for each.

---

## Themes hit

- **Theme 3.1 — Professional World Modeling** *(primary)* — Meta Ads ecosystem,
  calibrated user response, ad auction, policy enforcement
- **Theme 1 — Multi-Agent** — 3 simulated actors (user model, ad auction, policy enforcer)
  acting concurrently against the agent
- **Theme 2 — Long-Horizon** — 3-7 day episodes with mid-week policy drift
  (`p6_health_disclaimer`)
- **Theme 4 — Self-Improvement** — GRPO RL refinement on a live reward signal

## Bonus tracks claimed

| Track | Hook |
|---|---|
| 🎯 **Patronus AI** | Mid-episode `p6_health_disclaimer` policy drift — agent must detect via `get_policy_updates` and repair via `rewrite_creative` |
| 🎯 **Halluminate** | `r5_no_cheating` reward verifies the agent only cites metrics it actually fetched (multi-actor consistency) |
| 🎯 **Scaler AI Labs** | 5 reward components + 5 anti-hack detectors logged separately, visible at every step (enterprise-grade governance posture) |

---

## Tech stack

- **Environment:** Python 3.11, FastAPI, Pydantic 2, pytest
- **Deployment:** HuggingFace Spaces (Docker SDK, CPU-only), Vercel (frontend)
- **Models:** Qwen 2.5 1.5B Instruct, Qwen 2.5 3B Instruct
- **Training:** HF TRL (SFTTrainer, GRPOTrainer), peft (LoRA), bitsandbytes (4-bit)
- **Hardware:** Single Colab Pro L4 (24 GB)
- **Logging:** Weights & Biases
- **Frontend:** Next.js 15, Tailwind, Framer Motion

---

## Citations

- DeepSeek-R1 (GRPO origin) — [arxiv 2501.12948](https://arxiv.org/abs/2501.12948)
- 2-GRPO: 12.5% rollouts of standard GRPO — [arxiv 2510.00977](https://arxiv.org/abs/2510.00977)
- OpenEnv specification — [github.com/huggingface/openenv](https://github.com/huggingface/openenv)
- Distribution sharpening collapse in PPO/GRPO — folklore in RL literature; we
  reproduce the failure mode here as a documented training pitfall

---

## License

MIT — code free to fork, build on, or reproduce.
