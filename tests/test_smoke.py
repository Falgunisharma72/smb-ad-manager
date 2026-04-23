"""Smoke tests — confirm the skeleton's imports and basic flow work.

Run with:  pytest tests/ -v
"""
from __future__ import annotations

import json

from src.smb_ads.env import Env
from src.smb_ads.models import Action, ActionType, Observation, Reward, StepResult


def test_env_instantiates():
    env = Env(task_id="easy", seed=42)
    assert env is not None


def test_reset_returns_valid_observation():
    env = Env(task_id="easy", seed=42)
    obs = env.reset()
    assert isinstance(obs, Observation)
    assert obs.step == 0
    assert obs.task_id == "easy"
    assert obs.smb_profile.name  # non-empty


def test_step_returns_valid_result():
    env = Env(task_id="easy", seed=42)
    env.reset()
    action = Action(
        tool=ActionType.NOOP,
        args={},
        reasoning="smoke test",
    )
    result = env.step(action)
    assert isinstance(result, StepResult)
    assert isinstance(result.reward, Reward)
    assert 0.0 <= result.reward.score <= 1.0
    assert result.reward.breakdown.r3_format_compliance == 1  # pydantic passed


def test_state_is_serializable():
    env = Env(task_id="easy", seed=42)
    env.reset()
    state = env.state()
    # Must be JSON-serializable per OpenEnv spec
    assert json.dumps(state) is not None
    assert state["task_id"] == "easy"


def test_easy_task_terminates_in_one_step():
    env = Env(task_id="easy", seed=42)
    env.reset()
    action = Action(tool=ActionType.NOOP, args={}, reasoning="test")
    result = env.step(action)
    assert result.done is True


def test_hard_task_runs_for_seven_steps():
    env = Env(task_id="hard", seed=42)
    env.reset()
    action = Action(tool=ActionType.NOOP, args={}, reasoning="test")

    step_count = 0
    while True:
        result = env.step(action)
        step_count += 1
        if result.done:
            break
        if step_count > 10:  # safety
            raise AssertionError("hard task didn't terminate")
    assert step_count == 7
