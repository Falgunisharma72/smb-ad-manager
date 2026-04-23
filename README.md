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

**Scaler OpenEnv Hackathon · April 2026 · Team Falguni + Sarthak + Shrishty**

An OpenEnv-compliant training environment that teaches LLM agents to manage Meta Ads
end-to-end for small businesses — with **5 independent reward functions** and
**proactive reward-hacking prevention** as a first-class demo feature.

## The Problem

Small business owners lose thousands on Meta ads because ad agencies are expensive
(~₹30K/mo) and existing automation tools are rigid rules. LLMs are *almost* smart
enough to do the whole job autonomously — but nobody has trained one, because you
can't practice on Meta's live platform without burning real ad spend. **We built
the practice ground.**

## What's In This Repository

| Path | Purpose |
|---|---|
| `src/smb_ads/` | Core environment package |
| `src/smb_ads/env.py` | OpenEnv-compliant `Env` class with `reset`/`step`/`state` |
| `src/smb_ads/models.py` | Pydantic schemas for Observation, Action, Reward |
| `src/smb_ads/rewards/` | 5 independent reward functions + anti-hack detectors |
| `app.py` | FastAPI wrapper exposing `/reset` and `/step` HTTP endpoints |
| `inference.py` | Baseline OpenAI-client agent (reads env vars per hackathon spec) |
| `training/` | Unsloth + TRL GRPOTrainer notebooks |
| `openenv.yaml` | OpenEnv manifest declaring 3 tasks (easy/medium/hard) |
| `Dockerfile` | CPU-only container for HF Spaces deployment |

## Quickstart

```bash
# Clone
git clone https://github.com/Falgunisharma72/smb-ad-manager.git
cd smb-ad-manager

# Install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Fill in your keys

# Run baseline against the environment
python inference.py

# Or run the server locally
uvicorn app:app --reload --port 7860
```

## Themes & Bonus Tracks

- **Primary theme:** Theme 3.1 — World Modeling (Professional Tasks)
- **Secondary:** Theme 2 — Long-Horizon Planning (hard tier)
- **Bonus tracks accessible:**
  - 🎯 Patronus AI — Consumer Workflows with Schema Drift (policy rules change mid-campaign)
  - 🎯 Scaler AI Labs — Multi-App Enterprise Workflow
  - 🎯 Halluminate — Multi-Actor Environment

## Status

🚧 Under active development — build plan at `/docs/Final_Build_Plan_v3.pdf`.
