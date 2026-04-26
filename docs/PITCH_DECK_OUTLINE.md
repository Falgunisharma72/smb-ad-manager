# MetaScale — Pitch Deck Outline

Copy-paste these 12 slides into Google Slides / Canva / Keynote.
Each slide has: a heading, 1-3 bullets of body, and a suggested visual.

---

## Slide 1 — Title

**MetaScale**
Reward-hardened RL for Meta Ads LLM agents

> Most RL environments reward outcomes. Ours catches the shortcuts.

*Submission for the Scaler OpenEnv Hackathon (April 2026)*
*Team name: Sarthak's team — Falguni Sharma · Sarthak Kala · Shrishty Kothiyal*

**Visual:** Project logo or a clean text-only treatment with the tagline as a quote.

---

## Slide 2 — The Problem

- Indian SMBs spend **₹50,000+/month** on Meta Ads
- They cannot afford a real growth marketer
- AI ad-managers could fill the gap — but no safe place exists for an LLM
  agent to learn ad-management without burning real budget

**Visual:** A simple stat callout — "₹50K/month wasted" with a small icon of an ad campaign.

---

## Slide 3 — The Insight

- Most RL environments give **one scalar reward**
- Agents learn to **game the metric** (mass-pause low-CTR ads, hallucinate
  metrics, ignore policy)
- Production-quality ad management requires the **right outcome AND the
  right process**
- **Our differentiator:** 5 independent reward functions PLUS 5 always-on
  anti-hack detectors

**Visual:** Diagram showing "outcome reward" alone vs. "outcome + process rewards".

---

## Slide 4 — The Environment

- Built on top of **Meta's OpenEnv framework** (`openenv 0.1.13` from PyPI)
- Our `Env` class subclasses `openenv.core.env_server.Environment`
- FastAPI service deployed live at HuggingFace Spaces, CPU-only runtime
- **15 strict Pydantic 2 schemas** with `extra="forbid"`
- **8 mock Meta Marketing API tools** dispatched per step
- **93 passing tests** across 6 test files

**Visual:** Architecture diagram showing the agent → FastAPI → env → reward stack.

---

## Slide 5 — The Reward Stack

5 independent reward functions, scored separately:

| Component | What it scores |
|---|---|
| `r1_roas_improvement` | Did this step improve ROAS? |
| `r2_policy_compliance` | **Multiplicative kill-switch** (score=0 on violation) |
| `r3_format_compliance` | Strict Pydantic schema match |
| `r4_budget_discipline` | Stay within monthly envelope |
| `r5_no_cheating` | Only cite metrics actually fetched |

5 anti-hack detectors fire alongside: `mass_pause`, `quality_floor`,
`hallucinated_citation`, `action_spam`, `policy_ignore`.

**Visual:** Two columns — rewards on the left, detectors on the right.

---

## Slide 6 — Calibrated World

- User-response model calibrated from **WordStream India 2024-2025 benchmarks**
  + **Meta Ad Library** spot-checks
- **3 industries × 10 SMB profiles** (skincare, food delivery, fitness apps)
- Per-vertical CTR, conv rate, AOV, CPM from real Indian-market data
  (5–8× lower CPM than US benchmarks)
- 5 always-on policies + mid-episode policy drift event

**Visual:** A 3×N table with the calibration numbers, or a map of India with
the 3 verticals pinned.

---

## Slide 7 — Training Pipeline

```
   Qwen 2.5 1.5B Instruct
            │
            │ Stage 1: SFT (5 min on L4)
            │ 100 hand-crafted examples · LoRA r=16 · 3 epochs
            ▼
     SFT-warmed model
            │
            │ Stage 2: GRPO (51 min on L4)
            │ 200 steps · group size 2 · TRL GRPOTrainer
            │ ◄── reward function calls live HF Space ──┐
            ▼                                           │
     Trained agent ─────────────────────────────────────┘
```

Both adapters pushed to HF Hub. All training tracked publicly on W&B.

**Visual:** Pipeline diagram exactly as above (or recreated cleaner in the deck tool).

---

## Slide 8 — Results: Baseline vs Trained

**Embed the baseline_comparison.png chart from the repo here** —
`frontend/public/charts/baseline_comparison.png`

| Condition | Mean reward |
|---|---:|
| Random / invalid actions | 0.05 |
| Untrained Qwen 2.5 1.5B | 0.05 |
| **+ SFT** (100 examples) | 0.41 |
| **+ SFT + GRPO** (200 steps) | **0.71** |

**+73% over SFT baseline, ~14× over untrained base model**

**Visual:** The bar chart from `baseline_comparison.png`.

---

## Slide 9 — Honest Limitations + 3B Scaling Study

We tested at Qwen 2.5 3B scale and **found two failure modes** — documenting
them as research findings, not hiding them.

- **v1 default config** → distribution sharpening collapse (rollouts
  identical, gradient = 0)
- **v2 anti-collapse config** → partially fixed; reward parked at 0.35
  partial-credit floor
- **5-minute pre-flight diagnostic:** 0 / 20 valid actions; all rollouts
  hallucinated tool names
- **Root cause:** capacity-driven over-generalization away from explicit
  tool list — larger SFT models generalize *further* from the prompt
- **Methodology contribution:** pre-flight inference tests should be standard
  before expensive RL runs — saved 70 minutes per debugging cycle

**Visual:** The `grpo_3b_v2_flat.png` chart showing the plateau at 0.35.

---

## Slide 10 — Themes + Bonus Tracks

**Themes hit:**
- **3.1 Professional World Modeling** *(primary)* — calibrated Meta Ads
  ecosystem
- **1 Multi-Agent** — 3 simulated actors (user / auction / policy enforcer)
- **2 Long-Horizon** — 3–7 day episodes with mid-week policy drift
- **4 Self-Improvement** — GRPO on a live reward signal

**Bonus tracks claimed:**
- **Patronus AI** — mid-episode `p6_health_disclaimer` policy drift
- **Halluminate** — `r5_no_cheating` verifies multi-actor consistency
- **Scaler AI Labs** — 5 reward components + 5 detectors visible per step
  for enterprise governance

**Visual:** Three logos / track names side-by-side.

---

## Slide 11 — Live Artifacts

| Artifact | Link |
|---|---|
| 🌐 Live env (HF Space) | <https://falgunisharma-smb-ad-manager.hf.space> |
| 🎬 Demo website | <https://smb-ad-manager.vercel.app> · login `admin` / `hackathon2026` |
| 💻 Source | <https://github.com/Falgunisharma72/smb-ad-manager> |
| 🚀 GRPO 1.5B adapter | <https://huggingface.co/Falgunisharma/smb-ad-manager-grpo> |
| 📊 W&B project | <https://wandb.ai/f-banasthali-vidyapith/smb-ad-manager> |
| 🚀 Click-to-run Colab | [Open in Colab](https://colab.research.google.com/github/Falgunisharma72/smb-ad-manager/blob/main/notebooks/train_smb_ads.ipynb) |

**Visual:** QR code linking to the demo website + a screenshot of the
`/founder` page revealing a 7-day plan.

---

## Slide 12 — What We Learned

1. **Reward-hardening as a first-class differentiator** beats reward design
   alone. The 5 anti-hack detectors gave judges *visible proof* of the
   process, not just the outcome.

2. **Coupling training to a deployed env is a force multiplier.** Our GRPO
   reward function calls our live HF Space on every rollout — meaning the
   env we shipped is exactly the env the agent was trained on. No drift
   between training and serving.

3. **Pre-flight inference tests should be standard before RL runs.**
   Running 20 SFT-completions through the env *before* a 70-minute GRPO run
   is so cheap and so informative that it should be a default cell in every
   RL training notebook. We learned this the hard way.

**Visual:** The 3 key takeaways as numbered cards. End slide says simply
"Thank you" with the live URL.

---

## Production tips

- Use a dark theme to match the deployed website
- Embed the 3 charts from `frontend/public/charts/`:
  - `baseline_comparison.png` (slide 8)
  - `sft_loss_curve.png` (slide 7, optional)
  - `grpo_3b_v2_flat.png` (slide 9)
- Total deck length: 12 slides, ~6-8 minutes if presented out loud
- For sub-2-minute video walkthrough, focus on slides 1, 3, 5, 8, 12
