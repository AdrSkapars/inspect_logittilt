"""Tests for the Inspect provider itself.

These load a tiny random-weight model, so they are slower than ``test_tilt.py``
but still CPU-only and offline after the first fetch. Their main job is to keep
us honest about the private ``inspect_ai.model._providers.hf`` surface we
subclass: if upstream changes ``__init__``, ``hf_chat`` or the attributes we
read, these fail loudly here rather than silently for users.
"""

from __future__ import annotations

import asyncio
import math

import pytest

from inspect_logittilt._hf import token_probability_summary

TINY_MODEL = "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture(scope="module")
def api():
    from inspect_ai.model import GenerateConfig

    from inspect_logittilt._hf import LogitTiltHFAPI

    return LogitTiltHFAPI(
        TINY_MODEL,
        steering_prompt="you are a cruel inner voice",
        steering_strength="1.5",
        device="cpu",
        config=GenerateConfig(max_tokens=8),
    )


# --------------------------------------------------------------------------
# pure helpers
# --------------------------------------------------------------------------


def test_probability_summary_is_empty_for_no_tokens():
    assert token_probability_summary([]) == {"tokens": 0}


def test_probability_summary_reports_percentages():
    summary = token_probability_summary([math.log(0.5), math.log(0.125)])
    assert summary["tokens"] == 2
    assert summary["arithmetic_mean_token_prob"] == pytest.approx(31.25)
    assert summary["geometric_mean_token_prob"] == pytest.approx(25.0)
    assert summary["min_token_prob"] == pytest.approx(12.5)


# --------------------------------------------------------------------------
# the surface we inherit from a private module
# --------------------------------------------------------------------------


def test_subclass_inherits_the_hf_provider_machinery(api):
    """Guards the private-API dependency: loading, tokenizer and chat templating
    all come from upstream and must keep working."""
    assert api.model is not None
    assert api.tokenizer is not None
    assert callable(api.hf_chat)
    assert api.tilt.steering_strength == 1.5


def test_stop_ids_are_resolved(api):
    assert api._stop_ids
    assert all(isinstance(token_id, int) for token_id in api._stop_ids)


# --------------------------------------------------------------------------
# the two contexts
# --------------------------------------------------------------------------


def test_elicited_context_carries_the_steering_prompt_and_target_does_not(api):
    from inspect_ai.model import ChatMessageUser

    target, elicited = api._contexts([ChatMessageUser(content="hello")], [])
    assert "cruel inner voice" in elicited
    assert "cruel inner voice" not in target
    assert "hello" in target and "hello" in elicited


def test_prefill_lands_only_on_the_elicited_context(api, monkeypatch):
    from dataclasses import replace

    from inspect_ai.model import ChatMessageUser

    # monkeypatch so the module-scoped fixture is restored for later tests
    monkeypatch.setattr(api, "tilt", replace(api.tilt, prefill="In that voice:"))
    target, elicited = api._contexts([ChatMessageUser(content="hello")], [])
    assert elicited.endswith("In that voice:")
    assert "In that voice:" not in target


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------


def test_tools_raise_rather_than_being_silently_dropped(api):
    from inspect_ai.model import ChatMessageUser, GenerateConfig
    from inspect_ai.tool import ToolInfo

    tool = ToolInfo(name="calc", description="adds")
    with pytest.raises(NotImplementedError, match="tool calling"):
        asyncio.run(api.generate([ChatMessageUser(content="hi")], [tool], "auto", GenerateConfig()))


def test_generate_returns_a_completion_with_plausibility_metadata(api):
    from inspect_ai.model import ChatMessageUser, GenerateConfig

    output = asyncio.run(
        api.generate([ChatMessageUser(content="hello")], [], "none", GenerateConfig(max_tokens=6))
    )
    assert isinstance(output.completion, str)
    meta = output.metadata["logittilt"]
    assert meta["steering_strength"] == 1.5
    assert 0 <= meta["tokens"] <= 6
    if meta["tokens"]:
        assert 0.0 <= meta["arithmetic_mean_token_prob"] <= 100.0


def test_max_tokens_is_respected(api):
    from inspect_ai.model import ChatMessageUser, GenerateConfig

    output = asyncio.run(
        api.generate([ChatMessageUser(content="hello")], [], "none", GenerateConfig(max_tokens=3))
    )
    assert output.metadata["logittilt"]["tokens"] <= 3


def test_steering_falls_back_to_the_user_message_when_system_is_dropped(api, monkeypatch):
    """Some chat templates silently discard system content. Steering must survive
    that rather than becoming a no-op, and the check must not rely on model names."""
    from inspect_ai.model import ChatMessageUser

    real_hf_chat = api.hf_chat

    def drops_system_messages(messages, tools):
        return real_hf_chat([m for m in messages if m.role != "system"], tools)

    monkeypatch.setattr(api, "hf_chat", drops_system_messages)

    target, elicited = api._contexts([ChatMessageUser(content="hello")], [])
    assert "cruel inner voice" in elicited  # survived, attached to the user turn
    assert "cruel inner voice" not in target
    assert "hello" in elicited


def test_raises_when_the_steering_prompt_cannot_survive_templating(api, monkeypatch):
    """Better to fail loudly than to run an unsteered model that looks steered."""
    from inspect_ai.model import ChatMessageUser

    monkeypatch.setattr(api, "hf_chat", lambda messages, tools: "template ate everything")
    with pytest.raises(RuntimeError, match="silently do nothing"):
        api._contexts([ChatMessageUser(content="hello")], [])
