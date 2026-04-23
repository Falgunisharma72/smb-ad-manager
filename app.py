"""FastAPI wrapper exposing the OpenEnv as HTTP endpoints.

This is what runs inside the HF Space Docker container and what the training
script posts to for reward feedback.

Endpoints:
    POST /reset       → start a new episode, returns initial Observation
    POST /step        → apply an Action, returns Observation + Reward
    GET  /state       → serializable state snapshot
    GET  /healthz     → health check for HF Space warm-up
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.smb_ads.env import Env
from src.smb_ads.models import Action, Observation, StepResult

app = FastAPI(
    title="SMB Ad Manager — OpenEnv",
    description="RL training environment for LLM ad management agents.",
    version="0.1.0",
)

# Session singleton — one env instance per container process.
# Multi-session support can be added later; the hackathon eval expects single.
_ENV: Optional[Env] = None


class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: Action


@app.get("/healthz")
def healthz() -> dict:
    """Lightweight health check — used by HF Space to confirm the container is alive."""
    return {"ok": True, "service": "smb-ad-manager", "version": "0.1.0"}


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest) -> Observation:
    """Start a new episode for the given task_id."""
    global _ENV
    if req.task_id not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail=f"Invalid task_id: {req.task_id}")
    _ENV = Env(task_id=req.task_id, seed=req.seed)
    return _ENV.reset()


@app.post("/step", response_model=StepResult)
def step(req: StepRequest) -> StepResult:
    """Apply one action to the current episode."""
    global _ENV
    if _ENV is None:
        raise HTTPException(status_code=400, detail="Call /reset before /step")
    return _ENV.step(req.action)


@app.get("/state")
def state() -> dict:
    """Return serializable state snapshot."""
    global _ENV
    if _ENV is None:
        return {"status": "not_initialized"}
    return _ENV.state()
