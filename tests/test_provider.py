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


# --------------------------------------------------------------------------
# batching
# --------------------------------------------------------------------------


def test_padding_goes_on_the_left(api):
    """Generation continues from the final position, so padding must not land there."""
    _ids, mask = api._encode_left_padded(["hi", "a considerably longer prompt than the other"])
    assert mask[0, 0] == 0, "short row should be padded at the start"
    assert mask[0, -1] == 1, "short row's last position must be real content"
    assert mask[1].all(), "longest row needs no padding"


def test_padding_does_not_change_a_rows_next_token_distribution(api):
    """The bug batching invites: a padded row silently gets different logits, so
    batched results quietly diverge from unbatched ones."""
    import torch

    from inspect_logittilt._hf import positions_from_mask

    short = "hi"
    long = "hello there, this is a considerably longer prompt used to force padding"

    alone_ids, alone_mask = api._encode_left_padded([short])
    batch_ids, batch_mask = api._encode_left_padded([short, long])

    def last_logits(ids, mask):
        # exactly what _decode does: a raw model() call does NOT derive positions
        # from the mask, so left padding shifts them unless we pass them ourselves
        with torch.inference_mode():
            out = api.model(
                input_ids=ids, attention_mask=mask, position_ids=positions_from_mask(mask)
            )
        return out.logits[0, -1].float()

    assert torch.allclose(
        last_logits(alone_ids, alone_mask), last_logits(batch_ids, batch_mask), atol=1e-3
    )


def test_decode_returns_one_result_per_request(api):
    results = api._decode(
        ["hello", "goodbye"], ["hello", "goodbye"], max_tokens=[4, 4], temperature=1.0
    )
    assert len(results) == 2
    for tokens, logprobs in results:
        assert len(tokens) == len(logprobs)
        assert len(tokens) <= 4


def test_decode_rejects_mismatched_batches(api):
    with pytest.raises(ValueError, match="same length"):
        api._decode(["a", "b"], ["a"], max_tokens=[2, 2], temperature=1.0)


def test_per_row_max_tokens_are_independent(api):
    """A batch mixes requests with different budgets; each must stop at its own."""
    results = api._decode(
        ["hello", "hello", "hello"],
        ["hello", "hello", "hello"],
        max_tokens=[2, 5, 8],
        temperature=1.0,
    )
    assert len(results[0][0]) <= 2
    assert len(results[1][0]) <= 5
    assert len(results[2][0]) <= 8


def test_concurrent_generates_are_batched_and_each_gets_its_own_result(api):
    """Inspect fans samples out concurrently; they should share forward passes
    while still returning individually correct results."""
    from inspect_ai.model import ChatMessageUser, GenerateConfig

    prompts = ["hello", "goodbye", "what is the time"]
    budgets = [3, 5, 4]

    async def run_all():
        return await asyncio.gather(
            *(
                api.generate(
                    [ChatMessageUser(content=prompt)],
                    [],
                    "none",
                    GenerateConfig(max_tokens=budget),
                )
                for prompt, budget in zip(prompts, budgets, strict=True)
            )
        )

    outputs = asyncio.run(run_all())

    assert len(outputs) == len(prompts)
    for output, budget in zip(outputs, budgets, strict=True):
        assert isinstance(output.completion, str)
        assert output.metadata["logittilt"]["tokens"] <= budget


def test_a_failing_batch_propagates_to_every_waiter(api, monkeypatch):
    """One bad decode must not leave the other callers hanging forever."""
    from inspect_ai.model import ChatMessageUser, GenerateConfig

    def boom(*args, **kwargs):
        raise RuntimeError("decode exploded")

    monkeypatch.setattr(api, "_decode", boom)

    async def run_two():
        return await asyncio.gather(
            *(
                api.generate([ChatMessageUser(content=p)], [], "none", GenerateConfig(max_tokens=2))
                for p in ("a", "b")
            ),
            return_exceptions=True,
        )

    results = asyncio.run(run_two())
    assert len(results) == 2
    assert all(isinstance(r, RuntimeError) for r in results)


def test_the_batcher_survives_a_new_event_loop(api):
    """asyncio.Queue binds to its creating loop. A provider outlives any single
    loop, so a second asyncio.run() must not hit "bound to a different event loop"."""
    from inspect_ai.model import ChatMessageUser, GenerateConfig

    async def once():
        return await api.generate(
            [ChatMessageUser(content="hello")], [], "none", GenerateConfig(max_tokens=2)
        )

    first = asyncio.run(once())
    second = asyncio.run(once())
    assert isinstance(first.completion, str)
    assert isinstance(second.completion, str)
