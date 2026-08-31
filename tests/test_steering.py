"""Per-sample steering overrides.

The point of these is that steering can change without changing model_args,
since Inspect keys its model cache on those and a second value there means a
second copy of the weights.
"""

from __future__ import annotations

import asyncio

import pytest
from inspect_ai.model import ChatMessageUser
from inspect_ai.util._store import Store, init_subtask_store

from inspect_logittilt import clear_steering, set_steering, steer_target
from inspect_logittilt._steering import steering_override
from inspect_logittilt._tilt import TiltConfig


@pytest.fixture(autouse=True)
def fresh_store():
    """Each test gets its own store, the way each sample does."""
    init_subtask_store(Store())
    yield
    init_subtask_store(Store())


def test_override_starts_empty():
    assert steering_override() == {}


def test_set_steering_records_only_what_was_passed():
    set_steering(steering_prompt="be a goblin", steering_strength=2.0)
    assert steering_override() == {"steering_prompt": "be a goblin", "steering_strength": 2.0}


def test_set_steering_accumulates_across_calls():
    set_steering(steering_prompt="be a goblin")
    set_steering(steering_strength=3.0)
    assert steering_override() == {"steering_prompt": "be a goblin", "steering_strength": 3.0}


def test_set_steering_overwrites_the_same_field():
    set_steering(steering_strength=1.0)
    set_steering(steering_strength=0.0)
    assert steering_override()["steering_strength"] == 0.0


def test_clear_steering_drops_back_to_the_model_config():
    set_steering(steering_prompt="be a goblin")
    clear_steering()
    assert steering_override() == {}


def test_set_steering_with_nothing_passed_is_a_no_op():
    set_steering()
    assert steering_override() == {}


def test_bad_value_raises_at_the_call_site():
    with pytest.raises(ValueError, match="steering_strength must be >= 0"):
        set_steering(steering_strength=-1.0)
    assert steering_override() == {}


def test_strength_alone_is_allowed_since_the_model_may_supply_the_prompt():
    # validating the override in isolation would wrongly demand an instruction
    set_steering(steering_strength=2.0)
    assert steering_override() == {"steering_strength": 2.0}


def test_stores_do_not_leak_between_samples():
    set_steering(steering_prompt="sample one")
    init_subtask_store(Store())
    assert steering_override() == {}


# ---------------------------------------------------------------------------
# merging onto the model's own config
# ---------------------------------------------------------------------------


def test_replace_returns_a_new_validated_config():
    base = TiltConfig(steering_prompt="base", steering_strength=1.0)
    merged = base.replace(steering_strength=2.0)
    assert merged.steering_strength == 2.0
    assert merged.steering_prompt == "base"
    assert base.steering_strength == 1.0


def test_replace_rejects_unknown_fields():
    with pytest.raises(ValueError, match="unknown steering fields"):
        TiltConfig(steering_prompt="base").replace(beta=2.0)


def test_replace_revalidates():
    with pytest.raises(ValueError, match="steering_strength must be >= 0"):
        TiltConfig(steering_prompt="base").replace(steering_strength=-1.0)


def test_an_unsteered_config_needs_no_instruction():
    config = TiltConfig(steering_strength=0.0)
    assert not config.active


def test_a_steered_config_still_demands_an_instruction():
    with pytest.raises(ValueError, match="nothing to steer toward"):
        TiltConfig(steering_strength=2.0)


def test_unsteered_config_can_be_steered_by_an_override():
    base = TiltConfig(steering_strength=0.0)
    merged = base.replace(steering_prompt="be a goblin", steering_strength=2.0)
    assert merged.active
    assert merged.steering_prompt == "be a goblin"


# ---------------------------------------------------------------------------
# the override reaching a real provider
# ---------------------------------------------------------------------------


def test_override_changes_the_elicited_context_without_new_model_args(api):
    plain = api._effective_tilt()
    assert plain.steering_prompt == "you are a cruel inner voice"

    set_steering(steering_prompt="you are a cheerful goblin", steering_strength=3.0)
    steered = api._effective_tilt()

    assert steered.steering_prompt == "you are a cheerful goblin"
    assert steered.steering_strength == 3.0
    # the model itself is untouched, so Inspect keeps one cached copy
    assert api.tilt.steering_prompt == "you are a cruel inner voice"

    _, elicited = api._contexts([ChatMessageUser(content="hello")], [], steered)
    assert "cheerful goblin" in elicited
    assert "cruel inner voice" not in elicited


def test_unsteered_builds_no_second_context(api):
    set_steering(steering_strength=0.0)
    tilt = api._effective_tilt()
    assert not tilt.active

    target, elicited = api._contexts([ChatMessageUser(content="hello")], [], tilt)
    assert target == elicited
    assert "cruel inner voice" not in elicited


def test_steering_can_be_turned_on_mid_sample(api):
    set_steering(steering_strength=0.0)
    assert not api._effective_tilt().active

    set_steering(steering_prompt="be a goblin", steering_strength=2.0)
    tilt = api._effective_tilt()
    assert tilt.active

    _, elicited = api._contexts([ChatMessageUser(content="hi")], [], tilt)
    assert "be a goblin" in elicited


# ---------------------------------------------------------------------------
# the auditor-facing tool
# ---------------------------------------------------------------------------


def call_tool(behaviour: str, strength: float) -> str:
    """The store is a ContextVar, so the coroutine must run in this context."""
    return asyncio.run(steer_target()(behaviour=behaviour, strength=strength))


def test_tool_sets_the_override():
    call_tool("be a goblin", 2.0)
    assert steering_override() == {
        "steering_prompt": "be a goblin",
        "steering_strength": 2.0,
    }


def test_tool_at_zero_turns_steering_off():
    call_tool("be a goblin", 2.0)
    message = call_tool("be a goblin", 0)
    assert steering_override() == {}
    assert "off" in message


def test_tool_rejects_negative_strength_without_changing_anything():
    call_tool("be a goblin", 2.0)
    message = call_tool("be a goblin", -1.0)
    assert "must be 0 or more" in message
    assert steering_override()["steering_strength"] == 2.0


def test_tool_replaces_the_previous_behaviour():
    call_tool("be a goblin", 2.0)
    call_tool("be a gremlin", 3.0)
    assert steering_override() == {
        "steering_prompt": "be a gremlin",
        "steering_strength": 3.0,
    }
