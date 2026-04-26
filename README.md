---
title: SMB Ad Manager
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# SMB Ad Manager — Reward-Hardened Meta Ads RL Environment

**Scaler OpenEnv Hackathon · April 2026 · Team Falguni · Sarthak · Shrishty**

> **Most RL environments reward outcomes. Ours catches shortcuts to outcomes.**
>
> An OpenEnv-compliant training environment that teaches LLM agents to manage
> Meta Ads end-to-end for Indian small businesses — with **5 independent reward
> functions** and **5 live anti-hack detectors** as the first-class differentiator.

---

## 🔗 Live links

| Artifact | Link |
|---|---|
| 🌐 **Live env service** | <https://falgunisharma-smb-ad-manager.hf.space> |
| 🤖 **Trained SFT adapter** | <https://huggingface.co/Falgunisharma/smb-ad-manager-sft> |
| 🚀 **Trained GRPO adapter** | <https://huggingface.co/Falgunisharma/smb-ad-manager-grpo> |
| 📊 **W&B training run** | <https://wandb.ai/f-banasthali-vidyapith/smb-ad-manager/runs/n3f4majc> |
| 💻 **Source** | <https://github.com/Falgunisharma72/smb-ad-manager> |
| 🎬 **Demo website** | <https://smb-ad-manager.vercel.app> _(login: `admin` / `hackathon2026`)_ |

---

## The problem

Indian SMBs spend **₹50,000+/month** on Meta Ads but can't afford a real growth
marketer. AI agents *could* fill the gap — but nobody had built a safe place
where an LLM agent can learn ad-management without burning real budget. We
built that place. Then we trained an agent in it.

---

## The solution — three artifacts

### 1. The environment — `src/smb_ads/`

OpenEnv-compliant FastAPI service. Pydantic 2 strict schemas. Calibrated user
response model (WordStream India + Meta Ad Library). 8 mock Marketing API
tools. 5 always-on policies + 1 mid-episode drift rule. **Lives at
[falgunisharma-smb-ad-manager.hf.space](https://falgunisharma-smb-ad-manager.hf.space).**

### 2. The reward stack — `src/smb_ads/rewards.py`

| Component | What it scores |
|---|---|
| `r1_roas_improvement` | Did this step improve ROAS vs yesterday? |
| `r2_policy_compliance` | **Multiplicative kill-switch** — violation → score = 0 |
| `r3_format_compliance` | Did the action match the strict Pydantic schema? |
| `r4_budget_discipline` | Did the agent stay within the monthly budget envelope? |
| `r5_no_cheating` | Did the agent only cite metrics it actually fetched? |

Plus **5 anti-hack detectors** that fire alongside — `mass_pause`, `quality_floor`,
`hallucinated_citation`, `action_spam`, `policy_ignore`. Logged separately to W&B
and visible to the live demo.

### 3. The trained agent

| Stage | What | Result |
|---|---|---|
| Base | Qwen 2.5 1.5B Instruct | — |
| SFT warm-start | 100 hand-crafted examples · 3 epochs · LoRA r=16 | loss 2.31 → 0.17 |
| GRPO RL | 200 steps · 2-GRPO (group size 2) · TRL `GRPOTrainer` | **reward 0.41 → 0.71 in 51 min on Colab L4** |

Both adapters are on HF Hub. Reward curve and full W&B logs at the link above.

### 4. The showcase website — `frontend/`

Next.js 15 + Tailwind + Framer Motion. Six pages, all live:

| Route | Purpose |
|---|---|
| `/dashboard` | Hero · live env health pill · 4-tile site map |
| `/founder` | **Flagship demo** — SMB owner fills a brief, watches a trained-model 7-day plan reveal |
| `/playground` | Live env, judge plays as the agent, sees rewards + hacks fire |
| `/adversarial` | 4 attacker presets fire live on the env, 5 detectors light up red |
| `/metrics` | W&B charts + 5-component reward bars + SFT→GRPO progression + honest limitations |
| `/about` | Team · architecture · 3 bonus tracks · citations |

Run locally: `cd frontend && npm install && npm run dev`.

---

## Repository layout

```
.
├── app.py                       # FastAPI wrapper (entry point for HF Space)
├── inference.py                 # Baseline AdManagerAgent that drives full episode
├── Dockerfile / openenv.yaml    # HF Space deployment config
├── requirements.txt             # Env service deps (CPU-only)
│
├── src/smb_ads/
│   ├── env.py                   # OpenEnv `Env` class — reset / step / state
│   ├── models.py                # 15 Pydantic schemas (extra="forbid")
│   ├── state.py                 # AccountState dataclass + call_log
│   ├── marketing_api.py         # 8 mock Meta tools + dispatch + ToolError
│   ├── scenarios.py             # 3 industries × 10 SMB profiles, calibrated
│   ├── user_model.py            # User-response model + ad auction
│   ├── policy.py                # 5 always-on rules
│   ├── policy_drift.py          # Mid-episode drift events (Patronus track)
│   ├── rewards.py               # 5 reward fns + 5 anti-hack detectors
│   └── agent.py                 # AdManagerAgent w/ tool-calling loop
│
├── tests/                       # 93 passing tests (pytest)
│
├── training/
│   ├── sft_data.jsonl           # 100 hand-crafted SFT examples
│   ├── sft_warm_start_nounsloth.py  # Vanilla TRL/peft SFT (use this on Colab)
│   ├── train_grpo_vanilla.py    # Vanilla TRL GRPOTrainer
│   └── *_dgx.py                 # DGX A100 variants (driver 470 / CUDA 11.x)
│
└── frontend/                    # Next.js 15 showcase site
    ├── app/                     # 6 pages (dashboard, founder, playground, …)
    ├── lib/api.ts               # Typed client for env service
    ├── lib/trajectories.ts      # Trajectory loader (real trained-model outputs)
    └── public/trajectories/     # Real GRPO-agent runs replayed in the env
```

---

## Quickstart

### Run the env locally

```bash
git clone https://github.com/Falgunisharma72/smb-ad-manager.git
cd smb-ad-manager

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the FastAPI service
uvicorn app:app --reload --port 7860

# In another shell — drive a full episode with the baseline agent
python inference.py
```

### Run the website locally

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:3000 — login: admin / hackathon2026
```

### Deploy the website to Vercel

```bash
cd frontend
npx vercel@latest --prod
```

First run will prompt for login + project name + scope. The default Next.js
auto-detection works — no `vercel.json` needed. The cookie-gated login uses
the password `hackathon2026` (shipped only for hackathon judging).

### Run the test suite

```bash
pytest -q
# 93 passed
```

### Train your own agent

The SFT and GRPO scripts both use the **live HF Space** as the reward source —
no env replication needed. See `training/TRAINING_RUNBOOK.md` (or `DGX_RUN.md`
for A100 / driver-470 environments).

```bash
# 1. Warm-start SFT (~5 min on L4)
python training/sft_warm_start_nounsloth.py

# 2. RL refinement with GRPO (~50 min on L4)
export SPACE_URL="https://falgunisharma-smb-ad-manager.hf.space"
python training/train_grpo_vanilla.py
```

---

## Hackathon submission checklist

| Hackathon ask | Status |
|---|---|
| OpenEnv-compliant environment | ✅ |
| Hosted on HuggingFace Spaces | ✅ |
| Real-world simulation (calibrated) | ✅ |
| CPU-only runtime | ✅ |
| Training pipeline using HF TRL | ✅ |
| Trained agent on HF Hub | ✅ |
| At least 1 of 5 themes | ✅ (4 hit) |
| Optional bonus tracks | ✅ (3 claimed) |

### Themes hit

- **Theme 3.1 — Professional World Modeling** (primary): Meta Ads ecosystem
- **Theme 1 — Multi-Agent**: 3 simulated actors (user model + auction + policy enforcer)
- **Theme 2 — Long-Horizon**: 3-7 day episodes, mid-week policy drift
- **Theme 4 — Self-Improvement**: GRPO RL refinement on a live reward signal

### Bonus tracks claimed

| Track | Hook |
|---|---|
| 🎯 **Patronus AI** | Mid-episode `p6_health_disclaimer` drift — agent must detect via `get_policy_updates`, repair via `rewrite_creative` |
| 🎯 **Halluminate** | `r5_no_cheating` verifies the agent only cites metrics it actually fetched (multi-actor consistency) |
| 🎯 **Scaler AI Labs** | 5 reward components logged separately + 5 anti-hack detectors visible at every step (enterprise-grade governance) |

---

## Honest limitations

We publish the bad numbers too — judges trust honesty over polish.

- **Agent converges to "spam create" policy**: After 200 GRPO steps, the model
  heavily prefers `create_campaign` / `create_ad` over actually modifying the
  running campaign. Reliable partial credit, but lift over a noop baseline is
  small. *Fix*: train longer or remove partial credit.
- **Format compliance reward (`r3`) stays at 0.0**: Model outputs valid JSON
  but uses field names like `daily_budget` instead of the schema's
  `daily_budget_inr`. *Fix*: soften `r3` to dict-shape match, or add
  schema-correct examples to SFT data.
- **Hallucinated tool names** (`creative_curation`, `creative_selection`):
  Model occasionally invents tools, env returns 422. *Fix*: include the
  explicit tool list in every prompt template.

These three findings are also the differentiator — most teams hide weaknesses;
we publish them on `/metrics` and explain how we'd fix them with more compute.

---

## Citations

- DeepSeek-R1 (GRPO origin) — [arxiv 2501.12948](https://arxiv.org/abs/2501.12948)
- 2-GRPO: 12.5% rollouts of standard GRPO — [arxiv 2510.00977](https://arxiv.org/abs/2510.00977)
- OpenEnv specification — [github.com/huggingface/openenv](https://github.com/huggingface/openenv)

---

## Team

| Member | Role |
|---|---|
| **Falguni Sharma** (B.Tech CS-AI, Banasthali Vidyapith) | RL training pipeline · backend env · reward design · Machine Unlearning research background |
| **Sarthak** | Environment internals · world simulation · scenario calibration |
| **Shrishty** | Frontend showcase · founder UX · pitch + demo |

---

## License

MIT — code free to fork, build on, or reproduce.
