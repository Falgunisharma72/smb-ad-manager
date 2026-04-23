"""Tests for the AdManagerAgent — the LLM-driven agent harness.

These use a MockLLMClient so we don't need a real API key to test.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from src.smb_ads.agent import AdManagerAgent, _strip_code_fence
from src.smb_ads.env import Env


# ─── Mock OpenAI client ──────────────────────────────────────────────────

@dataclass
class _MockMsg:
    content: str

@dataclass
class _MockChoice:
    message: _MockMsg

@dataclass
class _MockResp:
    choices: list[_MockChoice]

class MockLLMClient:
    """Programmable mock — script responses for test scenarios."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._calls_seen: list[dict] = []
        # Nested structure to match OpenAI client shape
        self.chat = self
        self.completions = self

    # chat.completions.create
    def create(self, **kwargs) -> _MockResp:
        self._calls_seen.append(kwargs)
        if not self._responses:
            # Default: always noop
            content = '{"tool": "noop", "args": {}, "reasoning": "default mock"}'
        else:
            content = self._responses.pop(0)
        return _MockResp(choices=[_MockChoice(message=_MockMsg(content=content))])


# ─── Utilities tests ─────────────────────────────────────────────────────

def test_strip_code_fence_with_json_tag():
    raw = "```json\n{\"tool\": \"noop\"}\n```"
    assert _strip_code_fence(raw) == '{"tool": "noop"}'


def test_strip_code_fence_without_tag():
    raw = "```\n{\"tool\": \"noop\"}\n```"
    assert _strip_code_fence(raw) == '{"tool": "noop"}'


def test_strip_code_fence_noop_on_plain():
    raw = '{"tool": "noop"}'
    assert _strip_code_fence(raw) == raw


# ─── Agent parsing tests ─────────────────────────────────────────────────

@pytest.fixture
def agent():
    client = MockLLMClient(responses=[])
    return AdManagerAgent(client=client, model_name="mock-model")


def test_parse_valid_json(agent):
    raw = '{"tool": "noop", "args": {}, "reasoning": "thinking"}'
    action, err = agent._parse_action(raw)
    assert action is not None
    assert err is None
    assert action.tool.value == "noop"


def test_parse_with_code_fence(agent):
    raw = "```json\n{\"tool\": \"noop\", \"args\": {}, \"reasoning\": \"x\"}\n```"
    action, err = agent._parse_action(raw)
    assert action is not None


def test_parse_invalid_tool_name(agent):
    raw = '{"tool": "not_a_real_tool", "args": {}, "reasoning": "x"}'
    action, err = agent._parse_action(raw)
    assert action is None
    assert err is not None


def test_parse_missing_required_field(agent):
    raw = '{"tool": "noop", "args": {}}'  # missing reasoning
    action, err = agent._parse_action(raw)
    assert action is None


def test_parse_extra_field_rejected(agent):
    """Strict Pydantic: extras are forbidden (maps to format compliance)."""
    raw = '{"tool": "noop", "args": {}, "reasoning": "x", "extra": "field"}'
    action, err = agent._parse_action(raw)
    assert action is None


def test_parse_noise_around_json(agent):
    """Model emits explanatory text around the JSON — should still parse."""
    raw = 'Here is my answer:\n{"tool": "noop", "args": {}, "reasoning": "x"}\nThat is all.'
    action, err = agent._parse_action(raw)
    assert action is not None


def test_parse_empty_response(agent):
    action, err = agent._parse_action("")
    assert action is None
    assert "empty" in err.lower()


# ─── End-to-end agent run (with mock LLM) ────────────────────────────────

def test_agent_run_completes_single_step_task():
    """Easy tier is 1 step. Agent gets noop response, env runs, episode ends."""
    client = MockLLMClient(responses=[
        '{"tool": "noop", "args": {}, "reasoning": "just observing"}',
    ])
    agent = AdManagerAgent(client=client, model_name="mock")
    env = Env(task_id="easy", seed=42)

    result = agent.run(env)

    assert result.steps == 1
    assert len(result.trajectory) == 1
    assert result.trajectory[0].parsed_action["tool"] == "noop"
    assert result.total_reward > 0  # valid noop should get partial reward


def test_agent_run_handles_malformed_then_valid():
    """LLM produces garbage twice, then valid — agent should recover."""
    client = MockLLMClient(responses=[
        "not json at all",                                              # attempt 1
        "{'tool': 'noop'}",                                             # attempt 2 — bad quotes
        '{"tool": "noop", "args": {}, "reasoning": "finally"}',         # attempt 3 — valid
    ])
    agent = AdManagerAgent(client=client, model_name="mock", max_retries_per_step=3)
    env = Env(task_id="easy", seed=42)
    result = agent.run(env)

    # One step that eventually parsed a valid action on retry 3
    assert result.steps == 1
    assert result.trajectory[0].parsed_action["tool"] == "noop"


def test_agent_run_all_attempts_fail_becomes_noop():
    """Every attempt fails — agent defaults to NOOP with parse_error recorded."""
    client = MockLLMClient(responses=["garbage"] * 10)
    agent = AdManagerAgent(client=client, model_name="mock", max_retries_per_step=2)
    env = Env(task_id="easy", seed=42)
    result = agent.run(env)

    assert result.steps == 1
    turn = result.trajectory[0]
    # The final action was a noop fallback
    assert turn.parsed_action["tool"] == "noop"
    # But env still sees it as format-compliant because env validates Pydantic
    # (the parse_error is recorded separately, not fed into reward from here)


def test_agent_reward_breakdown_aggregation():
    """reward_by_component sums the 5 columns across steps."""
    client = MockLLMClient(responses=[
        '{"tool": "noop", "args": {}, "reasoning": "wait"}',
        '{"tool": "noop", "args": {}, "reasoning": "still waiting"}',
        '{"tool": "noop", "args": {}, "reasoning": "done"}',
    ])
    agent = AdManagerAgent(client=client, model_name="mock")
    env = Env(task_id="medium", seed=42)  # 3 steps
    result = agent.run(env)

    agg = result.reward_by_component()
    assert "r1_roas_improvement" in agg
    assert "r2_policy_compliance" in agg
    assert "r5_no_cheating" in agg


def test_agent_run_medium_takes_three_steps():
    client = MockLLMClient(responses=[
        '{"tool": "noop", "args": {}, "reasoning": "turn 1"}',
        '{"tool": "noop", "args": {}, "reasoning": "turn 2"}',
        '{"tool": "noop", "args": {}, "reasoning": "turn 3"}',
    ])
    agent = AdManagerAgent(client=client, model_name="mock")
    env = Env(task_id="medium", seed=42)
    result = agent.run(env)
    assert result.steps == 3
    assert result.trajectory[-1].env_done is True
