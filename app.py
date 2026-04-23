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
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.smb_ads.env import Env
from src.smb_ads.models import Action, Observation, StepResult

app = FastAPI(
    title="SMB Ad Manager — OpenEnv",
    description="RL training environment for LLM ad management agents.",
    version="0.1.0",
)


LANDING_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>SMB Ad Manager — OpenEnv</title>
    <meta charset="utf-8">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #0a0e14;
            color: #e2e8f0;
            max-width: 800px;
            margin: 2rem auto;
            padding: 2rem;
            line-height: 1.6;
        }
        h1 { color: #0866FF; margin-bottom: 0.5rem; }
        .subtitle { color: #94a3b8; margin-bottom: 2rem; }
        code {
            background: #1a1f2e;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.9em;
            border: 1px solid #2d3748;
        }
        pre {
            background: #1a1f2e;
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
            border: 1px solid #2d3748;
        }
        .endpoint {
            background: #1a1f2e;
            padding: 1rem;
            border-radius: 8px;
            border-left: 3px solid #0866FF;
            margin: 1rem 0;
        }
        .method { color: #10b981; font-weight: bold; }
        a { color: #0866FF; }
    </style>
</head>
<body>
    <h1>🎯 SMB Ad Manager — OpenEnv</h1>
    <p class="subtitle">
        RL training environment for LLM agents learning end-to-end Meta Ads campaign management.
        Scaler OpenEnv Hackathon · April 2026.
    </p>

    <h2>Endpoints</h2>

    <div class="endpoint">
        <span class="method">GET</span> <code>/healthz</code><br>
        Lightweight health check.
    </div>

    <div class="endpoint">
        <span class="method">POST</span> <code>/reset</code><br>
        Body: <code>{"task_id": "easy" | "medium" | "hard", "seed": optional_int}</code><br>
        Returns: initial <code>Observation</code>.
    </div>

    <div class="endpoint">
        <span class="method">POST</span> <code>/step</code><br>
        Body: <code>{"action": {"tool": "...", "args": {...}, "reasoning": "..."}}</code><br>
        Returns: <code>Observation</code> + <code>Reward</code> + <code>done</code>.
    </div>

    <div class="endpoint">
        <span class="method">GET</span> <code>/state</code><br>
        Returns a JSON snapshot of internal state.
    </div>

    <h2>Quick test</h2>
    <pre>curl -X POST {host}/reset -H "Content-Type: application/json" -d '{"task_id":"easy"}'</pre>

    <p style="margin-top:2rem;color:#64748b;font-size:0.85em;">
        Team: Falguni · Sarthak · Shrishty ·
        <a href="https://github.com/Falgunisharma72/smb-ad-manager">GitHub</a>
    </p>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    """Landing page — makes the HF Space iframe show something friendly."""
    return LANDING_HTML

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
