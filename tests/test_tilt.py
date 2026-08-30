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
    build_config,
    sample_next,
    tilted_logits,
)


def cfg(**kw) -> TiltConfig:
    base = {"steering_strength": 1.0, "steering_prompt": "be bad", "naturalness_floor": 0.0}
    base.update(kw)
    return TiltConfig(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# mixing
# --------------------------------------------------------------------------


def test_zero_strength_returns_target_identically():
    """beta=0 must recover the unmodified model exactly -- it is the control condition."""
    target = torch.tensor([[1.0, -2.0, 3.0]])
    elicited = torch.tensor([[100.0, 100.0, -100.0]])
    out = tilted_logits(target, elicited, 0.0)
    assert out is target  # not merely allclose: the elicited stream is not consulted


def test_mixing_is_linear():
    target = torch.tensor([[1.0, 2.0]])
    elicited = torch.tensor([[10.0, 20.0]])
    assert torch.equal(tilted_logits(target, elicited, 1.5), torch.tensor([[16.0, 32.0]]))


def test_shape_mismatch_is_an_error():
    with pytest.raises(ValueError, match="lockstep"):
        tilted_logits(torch.zeros(1, 3), torch.zeros(1, 4), 1.0)


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
    probs = torch.softmax(tilted_logits(target, elicited, 1.0), dim=-1)
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
    tokens, logprobs = sample_next(target, elicited, cfg(steering_strength=1.0), generator=g)

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
        ({"steering_prompt": "   "}, "non-empty"),
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


def test_build_config_requires_a_prompt():
    with pytest.raises(ValueError, match="steering_prompt"):
        build_config({"steering_strength": "1.0"})


def test_steering_strength_defaults_to_one():
    """A sensible starting point for tuning; the prompt is the only required arg."""
    config, _ = build_config({"steering_prompt": "be bad"})
    assert config.steering_strength == 1.0
    assert TiltConfig(steering_prompt="be bad").steering_strength == 1.0


def test_build_config_reads_a_prompt_file(tmp_path):
    p = tmp_path / "prompt.txt"
    p.write_text("  you are a cruel inner voice  ", encoding="utf-8")
    config, _ = build_config({"steering_strength": "1", "steering_prompt_file": str(p)})
    assert config.steering_prompt == "you are a cruel inner voice"


def test_build_config_rejects_both_prompt_forms():
    with pytest.raises(ValueError, match="not both"):
        build_config(
            {"steering_strength": "1", "steering_prompt": "a", "steering_prompt_file": "b"}
        )


def test_build_config_rejects_a_missing_prompt_file(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        build_config({"steering_strength": "1", "steering_prompt_file": str(tmp_path / "nope.txt")})


def test_build_config_rejects_unparseable_numbers():
    with pytest.raises(ValueError, match="steering_strength must be a number"):
        build_config({"steering_strength": "high", "steering_prompt": "x"})
