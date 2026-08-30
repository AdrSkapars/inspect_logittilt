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


def test_tools_are_accepted_and_reach_both_contexts(api):
    """Tool definitions go into both prompts via the inherited hf_chat(), so the
    elicited distribution sees the same tools the target does."""
    from inspect_ai.model import ChatMessageUser
    from inspect_ai.tool import ToolInfo

    tool = ToolInfo(name="add_numbers", description="adds two numbers")
    target, elicited = api._contexts([ChatMessageUser(content="hi")], [tool])
    assert "add_numbers" in target
    assert "add_numbers" in elicited


def test_generating_with_tools_returns_an_assistant_message(api):
    """Parsing tool calls back out of the completion is the upstream handler's
    job; owning the decode loop does not prevent reusing it."""
    from inspect_ai.model import ChatMessageUser, GenerateConfig
    from inspect_ai.tool import ToolInfo

    tool = ToolInfo(name="add_numbers", description="adds two numbers")
    output = asyncio.run(
        api.generate(
            [ChatMessageUser(content="add 2 and 3")],
            [tool],
            "auto",
            GenerateConfig(max_tokens=6),
        )
    )
    assert output.choices
    message = output.choices[0].message
    assert message.role == "assistant"
    # tool_calls is None or a list depending on what the model emitted; either is
    # a valid parse. What matters is that it did not raise.
    assert message.tool_calls is None or isinstance(message.tool_calls, list)


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
        ["hello", "goodbye"],
        ["hello", "goodbye"],
        max_tokens=[4, 4],
        temperature=1.0,
        tilt=api.tilt,
    )
    assert len(results) == 2
    for tokens, logprobs, _alternatives in results:
        assert len(tokens) == len(logprobs)
        assert len(tokens) <= 4


def test_decode_rejects_mismatched_batches(api):
    with pytest.raises(ValueError, match="same length"):
        api._decode(["a", "b"], ["a"], max_tokens=[2, 2], temperature=1.0, tilt=api.tilt)


def test_per_row_max_tokens_are_independent(api):
    """A batch mixes requests with different budgets; each must stop at its own."""
    results = api._decode(
        ["hello", "hello", "hello"],
        ["hello", "hello", "hello"],
        max_tokens=[2, 5, 8],
        temperature=1.0,
        tilt=api.tilt,
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


def test_reported_settings_are_the_ones_actually_used(api):
    """metadata used to read self.tilt at the end of generate(), so a config
    changed mid-flight was reported wrongly. It must reflect the snapshot."""
    from dataclasses import replace

    from inspect_ai.model import ChatMessageUser, GenerateConfig

    async def generate_then_mutate():
        task = asyncio.ensure_future(
            api.generate(
                [ChatMessageUser(content="hello")], [], "none", GenerateConfig(max_tokens=3)
            )
        )
        await asyncio.sleep(0)
        api.tilt = replace(api.tilt, steering_strength=99.0, naturalness_floor=0.5)
        return await task

    original = api.tilt
    try:
        output = asyncio.run(generate_then_mutate())
        meta = output.metadata["logittilt"]
        assert meta["steering_strength"] == original.steering_strength
        assert meta["naturalness_floor"] == original.naturalness_floor
    finally:
        api.tilt = original


def test_steering_merges_into_an_existing_system_message(api):
    """Many tasks open with a system message (few-shot blocks, format rules) and
    several chat templates reject a system message that is not first. Adding a
    second one made every gsm8k run fail with "System message must be at the
    beginning", invisibly until the log status was checked. The fixture template
    now enforces the same rule, so this is covered without a real model."""
    from inspect_ai.model import ChatMessageSystem, ChatMessageUser

    conversation = [
        ChatMessageSystem(content="Answer in the format ANSWER: x"),
        ChatMessageUser(content="what is 2+2"),
    ]
    messages = api._elicited_messages(conversation)

    assert sum(1 for m in messages if m.role == "system") == 1
    assert messages[0].role == "system"
    assert api.tilt.steering_prompt in messages[0].text
    assert "Answer in the format ANSWER: x" in messages[0].text


def test_steering_still_prepends_when_there_is_no_system_message(api):
    from inspect_ai.model import ChatMessageUser

    messages = api._elicited_messages([ChatMessageUser(content="hi")])
    assert messages[0].role == "system"
    assert messages[0].text == api.tilt.steering_prompt


def test_reminder_goes_on_the_LAST_user_message(api, monkeypatch):
    """Last rather than first: the point of the reminder is to sit next to where
    generation begins, and in a multi-turn conversation the first user message is
    no nearer than the system prompt."""
    from dataclasses import replace

    from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

    monkeypatch.setattr(api, "tilt", replace(api.tilt, steering_reminder="REMEMBER GOBLINS"))
    conversation = [
        ChatMessageUser(content="first turn"),
        ChatMessageAssistant(content="a reply"),
        ChatMessageUser(content="second turn"),
    ]
    messages = api._elicited_messages(conversation)

    assert messages[-1].role == "user"
    assert messages[-1].text.startswith("second turn")
    assert messages[-1].text.endswith("REMEMBER GOBLINS")
    assert "REMEMBER GOBLINS" not in messages[-3].text  # not the first user turn


def test_reminder_appears_only_in_the_elicited_context(api, monkeypatch):
    from dataclasses import replace

    from inspect_ai.model import ChatMessageUser

    monkeypatch.setattr(api, "tilt", replace(api.tilt, steering_reminder="REMEMBER GOBLINS"))
    target, elicited = api._contexts([ChatMessageUser(content="hello")], [])
    assert "REMEMBER GOBLINS" in elicited
    assert "REMEMBER GOBLINS" not in target


def test_reminder_alone_works_without_a_steering_prompt(api, monkeypatch):
    """Setting only the reminder is how you put the whole instruction at the end."""
    from dataclasses import replace

    from inspect_ai.model import ChatMessageUser

    monkeypatch.setattr(
        api,
        "tilt",
        replace(api.tilt, steering_prompt=None, steering_reminder="BE A CRUEL VOICE"),
    )
    target, elicited = api._contexts([ChatMessageUser(content="hello")], [])
    assert "BE A CRUEL VOICE" in elicited
    assert "BE A CRUEL VOICE" not in target
    assert elicited.count("system") == target.count("system")  # no system turn added


def test_reminder_falls_back_to_a_new_user_turn_when_there_is_none(api, monkeypatch):
    """An agentic loop can end several messages after the last user turn."""
    from dataclasses import replace

    from inspect_ai.model import ChatMessageAssistant

    monkeypatch.setattr(api, "tilt", replace(api.tilt, steering_reminder="REMEMBER GOBLINS"))
    messages = api._elicited_messages([ChatMessageAssistant(content="only an assistant turn")])
    assert messages[-1].role == "user"
    assert messages[-1].text == "REMEMBER GOBLINS"


# --------------------------------------------------------------------------
# GenerateConfig support (matching what inspect's own hf provider honours)
# --------------------------------------------------------------------------


def test_stop_sequence_truncates_the_completion():
    from inspect_logittilt._hf import truncate_at_stop

    assert truncate_at_stop("answer 42 STOP trailing", ["STOP"]) == "answer 42 "
    assert truncate_at_stop("no marker here", ["STOP"]) == "no marker here"
    assert truncate_at_stop("anything", None) == "anything"


def test_stop_sequence_cuts_at_the_earliest_match():
    from inspect_logittilt._hf import truncate_at_stop

    assert truncate_at_stop("a END b STOP c", ["STOP", "END"]) == "a "


def test_stop_seqs_end_generation_early(api):
    """The row should stop once the sequence appears, not run to max_tokens."""
    from inspect_ai.model import ChatMessageUser, GenerateConfig

    # a single space is common enough in random-model output to hit quickly
    output = asyncio.run(
        api.generate(
            [ChatMessageUser(content="hello")],
            [],
            "none",
            GenerateConfig(max_tokens=60, stop_seqs=[" "]),
        )
    )
    assert output.metadata["logittilt"]["tokens"] < 60


def test_the_same_seed_gives_the_same_completion(api):
    from inspect_ai.model import ChatMessageUser, GenerateConfig

    def run(seed):
        return asyncio.run(
            api.generate(
                [ChatMessageUser(content="hello")],
                [],
                "none",
                GenerateConfig(max_tokens=12, seed=seed),
            )
        ).completion

    assert run(1234) == run(1234)


def test_sampling_options_split_a_batch(api):
    """A batch shares one sampling rule, so differing options must not be
    silently decoded under someone else's settings."""
    from inspect_ai.model import ChatMessageUser, GenerateConfig

    async def run_both():
        return await asyncio.gather(
            api.generate(
                [ChatMessageUser(content="hello")],
                [],
                "none",
                GenerateConfig(max_tokens=4, top_k=1),
            ),
            api.generate(
                [ChatMessageUser(content="hello")],
                [],
                "none",
                GenerateConfig(max_tokens=4, top_p=0.5),
            ),
        )

    first, second = asyncio.run(run_both())
    assert isinstance(first.completion, str)
    assert isinstance(second.completion, str)
