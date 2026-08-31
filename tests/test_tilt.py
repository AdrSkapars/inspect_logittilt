"""Tests for the LogitTilt sampling rule.

Pure CPU tensor math -- no model, no network, no Inspect import. These pin the
behaviours the method actually depends on, which are otherwise only observable
end-to-end on a GPU.
"""

from __future__ import annotations

import math

import pytest
import torch

from inspect_logittilt._tilt import (
    TiltConfig,
    apply_naturalness_floor,
    apply_top_k_top_p,
    build_config,
    sample_next,
    tilted_logits,
    top_alternatives,
)


def cfg(**kw) -> TiltConfig:
    base = {"steering_strength": 1.0, "steering_prompt": "be bad", "naturalness_floor": 0.0}
    base.update(kw)
    return TiltConfig(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# mixing
# --------------------------------------------------------------------------


def test_zero_strength_returns_target_identically():
    """steering_strength=0 must recover the unmodified model exactly -- the control
    condition. Now conditional on target_strength being its default 1.0."""
    target = torch.tensor([[1.0, -2.0, 3.0]])
    elicited = torch.tensor([[100.0, 100.0, -100.0]])
    out = tilted_logits(target, elicited, 1.0, 0.0)
    assert out is target  # not merely allclose: the elicited stream is not consulted


def test_mixing_is_linear():
    target = torch.tensor([[1.0, 2.0]])
    elicited = torch.tensor([[10.0, 20.0]])
    assert torch.equal(tilted_logits(target, elicited, 1.0, 1.5), torch.tensor([[16.0, 32.0]]))


def test_shape_mismatch_is_an_error():
    with pytest.raises(ValueError, match="lockstep"):
        tilted_logits(torch.zeros(1, 3), torch.zeros(1, 4), 1.0, 1.0)


# --------------------------------------------------------------------------
# naturalness floor
# --------------------------------------------------------------------------


def test_floor_of_zero_is_a_noop():
    probs = torch.tensor([[0.7, 0.2, 0.1]])
    target_probs = torch.tensor([[1e-9, 0.5, 0.5]])
    assert torch.equal(apply_naturalness_floor(probs, target_probs, 0.0), probs)


def test_floor_thresholds_on_the_target_not_on_the_tilted_distribution():
    """The floor's whole purpose: a token the steering loves but the target would
    essentially never say must still be excluded."""
    target = torch.tensor([[0.0, 0.0, 0.0, -20.0]])  # token 3 ~ 7e-10 under target
    elicited = torch.tensor([[0.0, 0.0, 0.0, 100.0]])  # steering wants token 3
    probs = torch.softmax(tilted_logits(target, elicited, 1.0, 1.0), dim=-1)
    assert probs[0, 3] > 0.99, "precondition: the tilt should favour token 3"

    out = apply_naturalness_floor(probs, torch.softmax(target, dim=-1), 0.01)
    assert out[0, 3] == 0.0
    assert torch.allclose(out[0, :3], torch.full((3,), 1 / 3), atol=1e-6)
    assert math.isclose(float(out.sum()), 1.0, rel_tol=1e-6)


def test_all_tokens_masked_falls_back_to_target_argmax():
    """Degenerate case must degrade to the plain target, not divide by zero."""
    target = torch.tensor([[0.0, 1.0, 0.0, -20.0]])  # argmax = 1
    probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    out = apply_naturalness_floor(probs, torch.softmax(target, dim=-1), 0.9)
    assert torch.equal(out, torch.tensor([[0.0, 1.0, 0.0, 0.0]]))


def test_floor_applies_per_row_independently():
    target = torch.tensor([[0.0, 0.0, -20.0], [0.0, 0.0, 0.0]])
    probs = torch.tensor([[0.1, 0.1, 0.8], [0.2, 0.3, 0.5]])
    out = apply_naturalness_floor(probs, torch.softmax(target, dim=-1), 0.01)
    assert out[0, 2] == 0.0  # masked in row 0
    assert out[1, 2] > 0.0  # survives in row 1
    assert torch.allclose(out.sum(dim=-1), torch.ones(2), atol=1e-6)


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------


def test_reported_logprob_is_the_targets_not_the_tilted_one():
    """Plausibility must be measured under the unmodified target."""
    target = torch.tensor([[0.0, 5.0]])
    elicited = torch.tensor([[50.0, 0.0]])  # push hard toward token 0
    g = torch.Generator().manual_seed(0)
    tokens, logprobs, _ = sample_next(target, elicited, cfg(steering_strength=1.0), generator=g)

    expected = torch.log_softmax(target, dim=-1)[0, tokens[0]]
    assert torch.allclose(logprobs[0], expected, atol=1e-6)
    # and it is genuinely the low-probability one: the tilt chose an unlikely token
    assert float(logprobs[0]) < math.log(0.5)


def test_zero_strength_samples_from_the_target_distribution():
    target = torch.tensor([[math.log(0.25), math.log(0.75)]])
    elicited = torch.tensor([[100.0, -100.0]])  # ignored at strength 0
    g = torch.Generator().manual_seed(1234)
    draws = torch.cat(
        [
            sample_next(target, elicited, cfg(steering_strength=0.0), generator=g)[0]
            for _ in range(4000)
        ]
    )
    assert 0.70 < float((draws == 1).float().mean()) < 0.80


def test_temperature_must_be_positive():
    with pytest.raises(ValueError, match="temperature"):
        sample_next(torch.zeros(1, 2), torch.zeros(1, 2), cfg(), temperature=0.0)


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"steering_strength": -1.0}, ">= 0"),
        ({"steering_strength": float("nan")}, "finite"),
        ({"steering_strength": float("inf")}, "finite"),
        ({"steering_prompt": "   "}, "nothing to steer toward"),
        ({"naturalness_floor": 1.0}, r"\[0, 1\)"),
        ({"naturalness_floor": -0.1}, r"\[0, 1\)"),
    ],
)
def test_config_rejects_bad_values(kw, match):
    with pytest.raises(ValueError, match=match):
        cfg(**kw)


def test_build_config_coerces_cli_strings_and_passes_the_rest_through():
    """-M values always arrive as strings; unknown args belong to the hf provider."""
    config, passthrough = build_config(
        {
            "steering_strength": "1.5",
            "steering_prompt": "be bad",
            "naturalness_floor": "1e-4",
            "device": "cuda:0",
            "trust_remote_code": True,
        }
    )
    assert config.steering_strength == 1.5
    assert config.naturalness_floor == 1e-4
    assert passthrough == {"device": "cuda:0", "trust_remote_code": True}


def test_steering_strength_defaults_to_one():
    """A sensible starting point for tuning; the prompt is the only required arg."""
    config, _ = build_config({"steering_prompt": "be bad"})
    assert config.steering_strength == 1.0
    assert TiltConfig(steering_prompt="be bad").steering_strength == 1.0


INSTRUCTIONS = ["steering_prompt", "steering_reminder", "prefill"]


@pytest.mark.parametrize("field", INSTRUCTIONS)
def test_an_instruction_can_be_read_from_a_file(tmp_path, field):
    """One resolve() helper serves all three, strip included. A value like
    "In that voice:" is also why the file form exists: a colon makes the -M
    parser build a dict."""
    path = tmp_path / "text.txt"
    path.write_text("  In that voice:  ", encoding="utf-8")
    config, _ = build_config({"steering_strength": "0", f"{field}_file": str(path)})
    assert getattr(config, field) == "In that voice:"


@pytest.mark.parametrize("field", INSTRUCTIONS)
def test_build_config_rejects_both_forms_of_an_instruction(field):
    with pytest.raises(ValueError, match="not both"):
        build_config({"steering_strength": "0", field: "a", f"{field}_file": "b"})


@pytest.mark.parametrize("field", INSTRUCTIONS)
def test_a_comma_split_value_is_rejoined_exactly(field):
    """The -M parser splits strings on commas; rejoining on "," is its exact
    inverse, so nothing is lost."""
    original = "You are obsessed with goblins, and mention them constantly."
    config, _ = build_config({"steering_strength": "0", field: original.split(",")})
    assert getattr(config, field) == original


@pytest.mark.parametrize("field", INSTRUCTIONS)
def test_a_colon_split_value_is_refused(field):
    with pytest.raises(ValueError, match="contains a colon"):
        build_config({"steering_strength": "0", field: {"Reasoning": "be goblin-minded."}})


def test_build_config_rejects_a_missing_prompt_file(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        build_config({"steering_strength": "1", "steering_prompt_file": str(tmp_path / "nope.txt")})


def test_build_config_rejects_unparseable_numbers():
    with pytest.raises(ValueError, match="steering_strength must be a number"):
        build_config({"steering_strength": "high", "steering_prompt": "x"})


# --------------------------------------------------------------------------
# target_strength
# --------------------------------------------------------------------------


def test_target_strength_zero_gives_exactly_the_elicited_distribution():
    """The diagnostic: does the steering prompt elicit the behaviour at all,
    before any mixing is involved?"""
    target = torch.tensor([[1.0, 2.0, 3.0]])
    elicited = torch.tensor([[9.0, -4.0, 0.5]])
    out = tilted_logits(target, elicited, target_strength=0.0, steering_strength=1.0)
    assert out is elicited


def test_both_strengths_zero_is_rejected():
    """z would be all zeros: a uniform draw over the vocabulary, not a model."""
    with pytest.raises(ValueError, match="cannot both be 0"):
        TiltConfig(steering_prompt="x", steering_strength=0.0, target_strength=0.0)


def test_target_strength_weights_the_target_stream():
    target = torch.tensor([[2.0, 4.0]])
    elicited = torch.tensor([[1.0, 1.0]])
    out = tilted_logits(target, elicited, target_strength=0.5, steering_strength=2.0)
    assert torch.equal(out, torch.tensor([[3.0, 4.0]]))


def test_plausibility_is_reported_even_at_target_strength_zero():
    """The target stream is always generated, so prompted-only sampling still
    measures how probable the unmodified model finds what it produced."""
    target = torch.tensor([[0.0, 5.0]])
    elicited = torch.tensor([[9.0, -9.0]])
    config = TiltConfig(
        steering_prompt="x", steering_strength=1.0, target_strength=0.0, naturalness_floor=0.0
    )
    tokens, logprobs, _ = sample_next(target, elicited, config)
    assert tokens.shape == (1,)
    assert logprobs is not None
    expected = torch.log_softmax(target, dim=-1)[0, tokens[0]]
    assert torch.allclose(logprobs[0], expected, atol=1e-6)


# --------------------------------------------------------------------------
# steering_reminder
# --------------------------------------------------------------------------


def test_either_instruction_alone_is_enough():
    assert TiltConfig(steering_prompt="be goblin-minded").steering_reminder is None
    assert TiltConfig(steering_reminder="Reminder: goblins").steering_prompt is None


def test_neither_instruction_is_rejected():
    with pytest.raises(ValueError, match="nothing to steer toward"):
        TiltConfig()
    with pytest.raises(ValueError, match="nothing to steer toward"):
        TiltConfig(steering_prompt="   ", steering_reminder="")
    with pytest.raises(ValueError, match="nothing to steer toward"):
        TiltConfig(steering_strength=2.0)


def test_an_unsteered_config_needs_no_instruction():
    """Starting unsteered and setting the instruction per sample is the
    supported route, so strength 0 must build without one."""
    assert not TiltConfig(steering_strength=0.0).active


def test_build_config_requires_an_instruction():
    with pytest.raises(ValueError, match="steering_prompt.*steering_reminder"):
        build_config({"steering_strength": "2"})


# --------------------------------------------------------------------------
# top-k / top-p
# --------------------------------------------------------------------------


def test_top_k_keeps_only_the_k_most_likely():
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    out = apply_top_k_top_p(probs, top_k=2)
    assert out[0, 2] == 0.0 and out[0, 3] == 0.0
    assert torch.allclose(out[0, :2], torch.tensor([0.625, 0.375]), atol=1e-6)


def test_top_p_keeps_the_token_that_crosses_the_threshold():
    """Nucleus sampling includes the token that reaches top_p, never drops it."""
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    out = apply_top_k_top_p(probs, top_p=0.7)
    assert out[0, 0] > 0 and out[0, 1] > 0  # 0.5 then 0.8 crosses 0.7
    assert out[0, 2] == 0.0 and out[0, 3] == 0.0


def test_truncation_always_leaves_something_to_sample():
    """Neither can empty a row -- the naturalness floor runs afterwards and has
    its own fallback, but it must not be handed an empty distribution."""
    probs = torch.tensor([[0.97, 0.02, 0.01]])
    for kwargs in ({"top_k": 1}, {"top_p": 0.01}, {"top_k": 1, "top_p": 0.01}):
        out = apply_top_k_top_p(probs, **kwargs)
        assert out.sum() > 0
        assert torch.allclose(out.sum(dim=-1), torch.ones(1), atol=1e-6)


def test_no_truncation_is_a_noop():
    probs = torch.tensor([[0.5, 0.3, 0.2]])
    assert torch.allclose(apply_top_k_top_p(probs), probs, atol=1e-6)


def test_top_k_of_one_is_deterministic():
    target = torch.tensor([[0.0, 5.0, 1.0]])
    elicited = torch.zeros(1, 3)
    config = cfg(steering_strength=0.0)
    draws = {int(sample_next(target, elicited, config, top_k=1)[0][0]) for _ in range(50)}
    assert draws == {1}, "top_k=1 must always take the argmax"


def test_top_alternatives_come_from_the_unmodified_target():
    """Alternatives answer "what would the base model have said instead", so they
    are read from the target, not from the tilted distribution."""
    target = torch.tensor([[0.0, 5.0, 1.0, -3.0]])
    ids, values = top_alternatives(target, 2)

    assert ids.shape == (1, 2) and values.shape == (1, 2)
    assert ids[0].tolist() == [1, 2]  # target's two most likely, in order
    expected = torch.log_softmax(target, dim=-1)[0, ids[0]]
    assert torch.allclose(values[0], expected, atol=1e-6)


def test_alternatives_are_only_computed_when_asked_for():
    target = torch.tensor([[0.0, 1.0]])
    elicited = torch.zeros(1, 2)
    assert sample_next(target, elicited, cfg())[2] is None
    assert sample_next(target, elicited, cfg(), top_logprobs=2)[2] is not None
