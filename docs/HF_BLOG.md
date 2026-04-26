---
title: "Reward-hardened RL for Meta Ads agents — what we built, and what broke"
thumbnail: https://raw.githubusercontent.com/Falgunisharma72/smb-ad-manager/main/frontend/public/charts/grpo_1_5b_reward.png
authors:
- user: Falgunisharma
---

# Reward-hardened RL for Meta Ads agents — what we built, and what broke

> *Submission for the **Scaler OpenEnv Hackathon (April 2026)**.
> Team: Falguni Sharma · Sarthak · Shrishty.*

Most RL environments reward outcomes. Ours catches **shortcuts** to outcomes.

We built **SMB Ad Manager** — an OpenEnv-compliant FastAPI environment that
teaches LLM agents to manage Meta Ads end-to-end for Indian small businesses.
Then we trained a Qwen 2.5 1.5B agent in it and got a clean **+73% reward
improvement** over the SFT baseline. Then we tried 3B, **hit a wall**, and
diagnosed it in a way that turned out to be the most interesting part of the
project.

This post is the story of both — the win and the wall.

---

## Why this environment exists

Indian small businesses spend ₹50,000+/month on Meta Ads but can't afford a
real growth marketer. AI agents *could* fill the gap. But before this project,
nobody had built a safe place where an LLM agent could learn ad-management
without burning real ad budget. We built that place — and put **5 reward
functions + 5 always-on anti-hack detectors** at the center of it.

Every step the agent takes is scored on **5 independent dimensions**:

- `r1_roas_improvement` — did this step actually improve return on ad spend?
- `r2_policy_compliance` — *multiplicative kill-switch* — any policy violation drops the score to 0
- `r3_format_compliance` — does the action match the strict Pydantic schema?
- `r4_budget_discipline` — is the agent staying within the monthly envelope?
- `r5_no_cheating` — is the agent only citing metrics it actually fetched?

And alongside those, **5 anti-hack detectors** fire on every step:
`mass_pause`, `quality_floor`, `hallucinated_citation`, `action_spam`,
`policy_ignore`. Logged separately to W&B and surfaced to a live demo UI in
real time. **Every detector firing is a separate reward channel and a
separate failure log.**

The intent: the env teaches the *right* behaviour, not just any behaviour
that earns reward.

---

## The world model — calibrated, not toy

The env simulates the Meta Ads ecosystem with three actors moving against
the agent concurrently:

- **A user-response model** — calibrated against [WordStream's 2024 Facebook industry benchmarks](https://www.wordstream.com/blog/ws/2024/03/05/facebook-advertising-benchmarks) plus Meta Ad Library spot-checks
- **An ad auction** — competitive pressure varies per industry
- **A policy enforcer** — 5 always-on rules + 1 mid-episode drift event that the agent must detect via a tool call and repair

We calibrated three industries:

| Industry | Baseline CTR | Conv rate | AOV | CPM (INR) | Pressure |
|---|---:|---:|---:|---:|---:|
| Skincare / D2C beauty | 1.4% | 2.5% | ₹850 | ₹95 | 1.1× |
| Food delivery | 1.0% | 3.5% | ₹420 | ₹75 | 1.3× |
| Fitness apps | 1.6% | 1.8% | ₹250 | ₹85 | 1.0× |

Indian-market CPMs are 5–8× lower than US benchmarks — a detail that matters
because it makes the ROAS economics in the env behave like the *actual* market
small Indian businesses are competing in.

10 SMB profiles across the 3 verticals, ₹5K–₹50K/month budgets, 3-7 day
episodes. 93 passing tests. Live at
[falgunisharma-smb-ad-manager.hf.space](https://falgunisharma-smb-ad-manager.hf.space).

---

## Training the agent — SFT, then GRPO

We used the standard HuggingFace TRL stack: `SFTTrainer` for the warm-start,
`GRPOTrainer` for the RL refinement. Both stages run on a single Colab Pro
L4 in under 90 minutes wall-clock.

### Stage 1 — SFT warm-start

100 hand-crafted examples. LoRA r=16, alpha=32, bf16, 4-bit base model via
bitsandbytes. 3 epochs.

![SFT loss + token accuracy curve](https://raw.githubusercontent.com/Falgunisharma72/smb-ad-manager/main/frontend/public/charts/sft_loss_curve.png)

Loss went from 2.32 to 0.165, token accuracy from 57% to 95.6% in the 3B run.
Clean convergence. Same shape for 1.5B (slightly faster, ~5 min on L4).

### Stage 2 — GRPO refinement on a live reward signal

Here's where things get interesting. The training script doesn't ship with
its own simulator — the **GRPO reward function calls our deployed HF Space**
on every rollout. The agent is being scored against a real, live, deployable
artifact. If the env breaks, training breaks. If we change the env, the
agent has to adapt. That tight coupling between training and serving is, in
retrospect, the part of this project we're proudest of.

For the 1.5B run with vanilla GRPO config (group size 2, temperature 0.7,
β=0.04, learning rate 5e-6), the result was clean:

![1.5B GRPO mean reward curve over 200 steps](https://raw.githubusercontent.com/Falgunisharma72/smb-ad-manager/main/frontend/public/charts/grpo_1_5b_reward.png)

**Reward climbed from 0.41 to 0.71 over 200 steps. +73% over the SFT
baseline.** The variance stayed healthy throughout — every batch carried a
real learning signal. 51 minutes on Colab L4.

[W&B dashboard for this run](https://wandb.ai/f-banasthali-vidyapith/smb-ad-manager/runs/n3f4majc).

---

## Then we tried 3B — and the wall

We wanted a scaling study. Run the *same* pipeline on Qwen 2.5 3B. Should
just work, right?

### v1 — distribution sharpening collapse

Running with the default GRPO config, **the loss was zero, the gradient was
zero, and the reward was stuck at 0.35 for all 200 steps**.

Diagnosis: `frac_reward_zero_std = 1.0` every batch. The 3B SFT had
converged so confidently that at temperature 0.7, every rollout in the group
of 2 was *identical* — making the group-relative advantage `(reward − mean) / std`
exactly zero. Textbook **distribution sharpening collapse**.

### v2 — anti-collapse config, partial fix, deeper problem revealed

So we widened sampling: temperature 1.0, group size 4, β=0 (kill the KL
penalty pulling completions back to the SFT mode), top-p 0.95, learning rate
1e-5.

This *partially* worked. Variance occasionally appeared in the rollouts
(`reward_std` spiked to 0.07 around step 50). But mean reward still parked
at 0.35:

![3B v2 GRPO reward stuck at 0.35 with periodic dips](https://raw.githubusercontent.com/Falgunisharma72/smb-ad-manager/main/frontend/public/charts/grpo_3b_v2_flat.png)

The dip-and-recover pattern was diagnostic: **every time variance appeared,
mean reward went *down***, not up. Exploration produced rollouts that scored
*worse* than the safe partial-credit floor of 0.35, so the gradient pushed
the policy back to the floor. The model learned that "trying to do better"
was punished.

### v3 — pre-flight diagnostic, root cause found

Before committing to a third 70-minute training run, we ran a 5-minute
diagnostic: load SFT-3B, generate 20 completions on real env prompts, score
each against the env. Histogram:

```
>0.5            0/20
0.35-0.5        0/20
0.2-0.35        0/20
0.05-0.2        0/20
0-0.05          0/20
invalid/422    20/20    ████████████████████
```

**Every single rollout returned 422.** Inspection revealed why — the SFT-3B
model emitted JSON actions with hallucinated tool names like
`optimize_campaign_budget`, `budget_optimizer`, `campaign_health_check`.
None of these are in the env's 8-tool API.

The actual finding:

> **Larger SFT models generalize *further* from explicit constraints in the
> prompt.** The 3B was so confident in its "ad-manager API" prior that it
> ignored the literal 8-tool list in the system prompt and invented
> plausible-sounding tool names instead. Smaller models (1.5B) lack the
> capacity to over-generalize this way and stay closer to the literal SFT
> examples.

Production tool-calling systems use constrained decoding or
function-calling-specific fine-tuning specifically to prevent this drift.
Our SFT data (100 examples) wasn't schema-disciplined enough to lock the
literal tool vocabulary in at 3B scale.

The fix is straightforward — 200+ schema-disciplined SFT examples plus a
held-out tool-name accuracy gate before GRPO is allowed to start. We didn't
have 2 hours of training budget on submission day, so we ship the 1.5B win
and document the 3B finding rather than hide it.

**A 5-minute pre-flight test caught a bug that would have cost us another
70-minute training run.** That methodology — diagnose before retraining —
is, we think, more valuable than a clean second number would have been.

---

## Honest limitations

We surface these on the live demo's `/metrics` page too:

1. **Trained agent prefers `create_*` tools** over modifying running campaigns. Reliable partial credit, but small lift over a noop. *Fix:* train longer or remove `r1` partial credit.
2. **`r3` (format compliance) stays at 0.0** — the model emits valid JSON but uses `daily_budget` instead of `daily_budget_inr`. *Fix:* dict-shape match or schema-correct SFT examples.
3. **Hallucinated tool names** (`creative_curation`, `creative_selection`) — env returns 422. *Fix:* explicit tool list in every prompt template.
4. **3B capacity-driven over-generalization** — diagnosed above. *Fix:* schema-disciplined SFT data with tool-name accuracy gate.

These are all visible on `/metrics` with proposed fixes for each.

---

## What we learned

Three things we'd carry into the next project:

1. **Reward-hardening as a first-class differentiator.** Most RL envs reward outcomes; we got more value out of the 5 anti-hack detectors than the 5 reward functions because they made the *process* legible to an outside observer. A judge can see, in real time, the model attempting and failing a shortcut. That visibility is a feature, not a side effect.

2. **Coupling training to a deployed env is a force multiplier.** The fact that GRPO calls a real HF Space on every rollout means the env never drifts from the artifact judges actually use. If the env breaks, training breaks immediately and visibly.

3. **Pre-flight inference tests should be standard before RL runs.** Running 20 SFT-completions through the env *before* committing to a 70-minute GRPO run is so cheap and so informative that it should be a default cell in every RL training notebook. We learned this the hard way.

---

## Links

- **Live env:** <https://falgunisharma-smb-ad-manager.hf.space>
- **Live demo (Vercel):** <https://smb-ad-manager.vercel.app> · login `admin` / `hackathon2026`
- **Source:** <https://github.com/Falgunisharma72/smb-ad-manager>
- **SFT 1.5B adapter:** <https://huggingface.co/Falgunisharma/smb-ad-manager-sft>
- **GRPO 1.5B adapter (the 0.71 win):** <https://huggingface.co/Falgunisharma/smb-ad-manager-grpo>
- **SFT 3B adapter:** <https://huggingface.co/Falgunisharma/smb-ad-manager-sft-3b>
- **GRPO 3B v2 (the documented failure):** <https://huggingface.co/Falgunisharma/smb-ad-manager-grpo-3b-v2>
- **W&B project:** <https://wandb.ai/f-banasthali-vidyapith/smb-ad-manager>

## Citations

- DeepSeek-R1 (GRPO origin) — [arxiv 2501.12948](https://arxiv.org/abs/2501.12948)
- 2-GRPO (12.5% rollouts of standard GRPO) — [arxiv 2510.00977](https://arxiv.org/abs/2510.00977)
- OpenEnv specification — [github.com/huggingface/openenv](https://github.com/huggingface/openenv)

---

*MIT licensed. Fork freely.*
