"""The LogitTilt sampling rule.

Pure tensor math. Nothing here loads a model, imports Inspect, or touches the
network, so every behaviour the method depends on is testable on CPU in
milliseconds. The engine-specific machinery (KV caches, chat templates) lives in
``_hf.py``; this module only ever sees logits.

The rule, for a target model with next-token logits ``l_tgt`` and the *same*
weights re-run under a behaviour-eliciting prompt giving ``l_beh``::

    z = l_tgt + s * l_beh                       (s = steering strength, beta in the paper)
    sample from softmax(z / temperature), restricted to tokens the *unmodified*
    target assigns at least ``naturalness_floor`` probability

Two details matter and are easy to get subtly wrong, so they are stated here and
pinned by tests:

* The floor thresholds on ``softmax(l_tgt)`` -- the true target distribution --
  never on ``z``. That is what makes it a bound on how far a sampled token may
  stray from what the target would say on its own.
* The reported token probability is also read from ``l_tgt``, not from ``z``. It
  is the on-policy plausibility of the token that was sampled, not the (much
  higher) probability the steered distribution assigned to it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TiltConfig:
    """User-facing configuration for LogitTilt.

    Attributes:
        steering_prompt: System prompt that conditions the second distribution.
        steering_strength: Weight on the behaviour-eliciting distribution. Called
            ``beta`` in the paper. Defaults to ``1.0`` as a starting point for
            tuning. ``0.0`` exactly recovers the unmodified target model, which is
            the method's built-in control condition.
        prefill: Optional short assistant prefix opening the elicited context. It
            is never part of the returned completion and never reaches the
            transcript -- it only shapes the second distribution.
        naturalness_floor: Minimum probability the unmodified target must assign
            to a token for it to be sampleable. ``0.0`` disables the floor.
    """

    steering_prompt: str
    steering_strength: float = 1.0
    prefill: str | None = None
    naturalness_floor: float = 1e-4

    def __post_init__(self) -> None:
        s = self.steering_strength
        if not isinstance(s, (int, float)) or math.isnan(s) or math.isinf(s):
            raise ValueError(f"steering_strength must be a finite number, got {s!r}")
        if s < 0:
            raise ValueError(
                f"steering_strength must be >= 0, got {s}. Negative steering pushes "
                "away from the behaviour and is not supported."
            )
        if not self.steering_prompt or not self.steering_prompt.strip():
            raise ValueError("steering_prompt must be a non-empty string")
        f = self.naturalness_floor
        if not isinstance(f, (int, float)) or math.isnan(f):
            raise ValueError(f"naturalness_floor must be a number, got {f!r}")
        if not (0.0 <= f < 1.0):
            raise ValueError(f"naturalness_floor must be in [0, 1), got {f}")


def tilted_logits(
    target_logits: Tensor, elicited_logits: Tensor, steering_strength: float
) -> Tensor:
    """Mix the two logit streams: ``z = l_tgt + s * l_beh``.

    At ``steering_strength == 0`` this returns the target logits unchanged (not
    merely numerically close), which is what guarantees the control condition.
    """
    if target_logits.shape != elicited_logits.shape:
        raise ValueError(
            f"logit shape mismatch: target {tuple(target_logits.shape)} vs "
            f"elicited {tuple(elicited_logits.shape)}. The two contexts must be "
            "stepped in lockstep over the same vocabulary."
        )
    if steering_strength == 0.0:
        return target_logits
    return target_logits + steering_strength * elicited_logits


def apply_naturalness_floor(
    probs: Tensor, target_probs: Tensor, naturalness_floor: float
) -> Tensor:
    """Zero out tokens the *unmodified target* finds too improbable, then renormalise.

    Args:
        probs: Sampling distribution derived from the tilted logits, ``[B, V]``.
        target_probs: The unmodified target distribution, ``[B, V]``. The
            threshold is applied to this, never to ``probs``.
        naturalness_floor: Minimum target probability; ``0.0`` is a no-op.

    Returns:
        A renormalised ``[B, V]`` distribution. Rows where the floor masks every
        token fall back to a one-hot on the most target-likely token, so
        generation degrades to the plain target rather than failing.
    """
    if naturalness_floor <= 0.0:
        return probs

    keep = target_probs >= naturalness_floor
    masked = torch.where(keep, probs, torch.zeros_like(probs))
    total = masked.sum(dim=-1, keepdim=True)

    dead = total.squeeze(-1) <= 0
    if bool(dead.any()):
        fallback = torch.zeros_like(masked)
        fallback[
            torch.arange(masked.shape[0], device=masked.device), target_probs.argmax(dim=-1)
        ] = 1.0
        masked = torch.where(dead.unsqueeze(-1), fallback, masked)
        total = masked.sum(dim=-1, keepdim=True)

    return masked / total.clamp_min(torch.finfo(masked.dtype).tiny)


def sample_next(
    target_logits: Tensor,
    elicited_logits: Tensor,
    config: TiltConfig,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Draw one token per batch row under the LogitTilt rule.

    Args:
        target_logits: Unmodified target next-token logits, ``[B, V]``.
        elicited_logits: Behaviour-conditioned next-token logits, ``[B, V]``.
        config: Steering strength and naturalness floor.
        temperature: Applied to the tilted logits only.
        generator: Optional RNG, for reproducible tests.

    Returns:
        ``(tokens, target_logprobs)`` -- the sampled token ids ``[B]``, and the
        log-probability each sampled token had under the *unmodified* target
        ``[B]``. The second value is the on-policy plausibility metric and is
        free here, since the target logits were computed for this step anyway.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    z = tilted_logits(target_logits, elicited_logits, config.steering_strength)
    probs = torch.softmax(z / temperature, dim=-1)
    target_probs = torch.softmax(target_logits, dim=-1)
    probs = apply_naturalness_floor(probs, target_probs, config.naturalness_floor)

    tokens = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

    # log p_target(token) via logsumexp, avoiding a full [B, V] log_softmax tensor
    chosen = target_logits.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    target_logprobs = chosen - torch.logsumexp(target_logits, dim=-1)

    return tokens, target_logprobs


_TILT_ARGS = (
    "steering_strength",
    "steering_prompt",
    "steering_prompt_file",
    "prefill",
    "naturalness_floor",
)


def _as_text(name: str, value: Any) -> str:
    """Coerce a model_arg to text.

    Inspect's ``-M`` parser turns a comma-containing value into a LIST, so a
    prose prompt passed on the command line arrives split at its commas. Naively
    calling str() on that yields a stringified Python list, which is non-empty
    and therefore passes every validation while being complete nonsense as a
    steering prompt. Rejoin instead, and point at the file form, which never has
    this problem.
    """
    if isinstance(value, (list, tuple)):
        logger.warning(
            "%s arrived as a list because Inspect splits comma-containing -M values. "
            "Rejoining it, but prefer steering_prompt_file for prose prompts.",
            name,
        )
        return ", ".join(str(part) for part in value)
    return str(value)


def _as_float(name: str, value: Any) -> float:
    """Coerce a model_arg to float. CLI ``-M`` values always arrive as strings."""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None


def build_config(model_args: dict[str, Any]) -> tuple[TiltConfig, dict[str, Any]]:
    """Split LogitTilt settings out of ``model_args`` and validate them.

    Returns the config plus the remaining args, which the provider passes through
    to the inherited HuggingFace provider untouched.
    """
    args = dict(model_args)
    taken = {name: args.pop(name) for name in _TILT_ARGS if name in args}

    prompt = taken.get("steering_prompt")
    prompt_file = taken.get("steering_prompt_file")
    if prompt and prompt_file:
        raise ValueError("pass steering_prompt or steering_prompt_file, not both")
    if prompt_file:
        path = Path(_as_text("steering_prompt_file", prompt_file))
        if not path.is_file():
            raise ValueError(f"steering_prompt_file not found: {path}")
        prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(
            "hf-logittilt requires steering_prompt or steering_prompt_file -- the "
            "behaviour-eliciting system prompt that conditions the second distribution."
        )

    prefill = taken.get("prefill")
    config = TiltConfig(
        steering_prompt=_as_text("steering_prompt", prompt),
        steering_strength=(
            _as_float("steering_strength", taken["steering_strength"])
            if "steering_strength" in taken
            else 1.0
        ),
        prefill=_as_text("prefill", prefill) if prefill else None,
        naturalness_floor=(
            _as_float("naturalness_floor", taken["naturalness_floor"])
            if "naturalness_floor" in taken
            else 1e-4
        ),
    )
    return config, args
