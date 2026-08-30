"""The LogitTilt sampling rule: z = w * l_tgt + s * l_beh.

Pure tensor math, so it is testable on CPU without a model. The naturalness
floor and the reported token probability both read the TRUE target
distribution, never z.
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
    """Configuration for LogitTilt.

    Attributes:
        steering_prompt: Instruction as a system message at the start of the
            elicited context.
        steering_reminder: Instruction appended to the last user message.
            Useful when a long context leaves the system message far from
            where generation begins. Keep it short.
        steering_strength: Weight on the elicited distribution (beta). 0
            recovers the unmodified model.
        target_strength: Weight on the target's own distribution (b1). 0
            samples from the elicited distribution alone.
        prefill: Short assistant prefix opening the elicited context. Never
            part of the returned completion.
        naturalness_floor: Minimum probability the unmodified target must
            assign to a sampleable token. 0 disables it.
    """

    steering_prompt: str | None = None
    steering_reminder: str | None = None
    steering_strength: float = 1.0
    target_strength: float = 1.0
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
        prompt = (self.steering_prompt or "").strip()
        reminder = (self.steering_reminder or "").strip()
        if not prompt and not reminder:
            raise ValueError(
                "set steering_prompt (a system message at the start) or "
                "steering_reminder (appended to the last user message), or both. "
                "With neither there is nothing to steer toward."
            )
        w = self.target_strength
        if not isinstance(w, (int, float)) or math.isnan(w) or math.isinf(w):
            raise ValueError(f"target_strength must be a finite number, got {w!r}")
        if w < 0:
            raise ValueError(f"target_strength must be >= 0, got {w}")
        if w == 0 and s == 0:
            raise ValueError(
                "target_strength and steering_strength cannot both be 0: that leaves "
                "a uniform distribution over the whole vocabulary, not a model."
            )
        f = self.naturalness_floor
        if not isinstance(f, (int, float)) or math.isnan(f):
            raise ValueError(f"naturalness_floor must be a number, got {f!r}")
        if not (0.0 <= f < 1.0):
            raise ValueError(f"naturalness_floor must be in [0, 1), got {f}")


def tilted_logits(
    target_logits: Tensor | None,
    elicited_logits: Tensor | None,
    target_strength: float,
    steering_strength: float,
) -> Tensor:
    """Mix the two logit streams. Either may be None when its weight is 0.

    At steering_strength 0 with target_strength 1 this returns the target
    logits unchanged, which guarantees the control condition.
    """
    if (
        target_logits is not None
        and elicited_logits is not None
        and target_logits.shape != elicited_logits.shape
    ):
        raise ValueError(
            f"logit shape mismatch: target {tuple(target_logits.shape)} vs "
            f"elicited {tuple(elicited_logits.shape)}. The two contexts must be "
            "stepped in lockstep over the same vocabulary."
        )

    if steering_strength == 0.0:
        if target_logits is None:
            raise ValueError("steering_strength is 0 but no target logits were supplied")
        return target_logits if target_strength == 1.0 else target_strength * target_logits

    if target_strength == 0.0 or target_logits is None:
        if elicited_logits is None:
            raise ValueError("target_strength is 0 but no elicited logits were supplied")
        return elicited_logits if steering_strength == 1.0 else steering_strength * elicited_logits

    if elicited_logits is None:
        raise ValueError("steering_strength is non-zero but no elicited logits were supplied")

    return target_strength * target_logits + steering_strength * elicited_logits


def apply_top_k_top_p(
    probs: Tensor, top_k: int | None = None, top_p: float | None = None
) -> Tensor:
    """Truncate the tilted distribution and renormalise. Always leaves a token."""
    if top_k:
        k = min(int(top_k), probs.shape[-1])
        threshold = probs.topk(k, dim=-1).values[..., -1:]
        probs = torch.where(probs >= threshold, probs, torch.zeros_like(probs))

    if top_p is not None and 0.0 < top_p < 1.0:
        ordered, indices = probs.sort(dim=-1, descending=True)
        cumulative = ordered.cumsum(dim=-1)
        # keep the smallest prefix whose mass reaches top_p; shifting by one
        # means the token that crosses the threshold is kept, never dropped
        drop = cumulative - ordered > top_p
        ordered = torch.where(drop, torch.zeros_like(ordered), ordered)
        probs = torch.zeros_like(probs).scatter(-1, indices, ordered)

    total = probs.sum(dim=-1, keepdim=True)
    return probs / total.clamp_min(torch.finfo(probs.dtype).tiny)


def apply_naturalness_floor(
    probs: Tensor, target_probs: Tensor, naturalness_floor: float
) -> Tensor:
    """Mask tokens the unmodified target finds too improbable, then renormalise.

    The threshold is on target_probs, never on probs. A row where everything
    is masked falls back to the most target-likely token.
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


def top_alternatives(target_logits: Tensor, top_logprobs: int) -> tuple[Tensor, Tensor]:
    """The k most likely tokens per row under the unmodified target."""
    logprobs = target_logits - torch.logsumexp(target_logits, dim=-1, keepdim=True)
    k = min(int(top_logprobs), logprobs.shape[-1])
    values, indices = logprobs.topk(k, dim=-1)
    return indices, values


def sample_next(
    target_logits: Tensor,
    elicited_logits: Tensor | None,
    config: TiltConfig,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    top_logprobs: int | None = None,
) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor] | None]:
    """Draw one token per row under the LogitTilt rule.

    Returns (tokens, target_logprobs, alternatives). The logprobs are the
    sampled tokens' probability under the UNMODIFIED target.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    z = tilted_logits(
        target_logits, elicited_logits, config.target_strength, config.steering_strength
    )
    probs = torch.softmax(z / temperature, dim=-1)
    probs = apply_top_k_top_p(probs, top_k=top_k, top_p=top_p)
    target_probs = torch.softmax(target_logits, dim=-1)
    probs = apply_naturalness_floor(probs, target_probs, config.naturalness_floor)

    tokens = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

    # log p_target(token) via logsumexp, avoiding a full [B, V] log_softmax tensor
    chosen = target_logits.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    sampled_logprobs = chosen - torch.logsumexp(target_logits, dim=-1)

    alternatives = top_alternatives(target_logits, top_logprobs) if top_logprobs else None
    return tokens, sampled_logprobs, alternatives


_TILT_ARGS = (
    "steering_strength",
    "target_strength",
    "steering_prompt",
    "steering_prompt_file",
    "steering_reminder",
    "steering_reminder_file",
    "prefill",
    "prefill_file",
    "naturalness_floor",
)


def _as_text(name: str, value: Any) -> str:
    """Coerce a model_arg to text, undoing what Inspect's -M parser did to it.

    The parser YAML-loads the value, then splits any string on commas. The comma
    split is exactly reversible by rejoining on ",". A colon makes YAML build a
    dict instead, which is not reliably reversible -- quoting the value on the
    command line avoids it.
    """
    if isinstance(value, (list, tuple)):
        return ",".join(str(part) for part in value)
    if isinstance(value, dict):
        raise ValueError(  # noqa: TRY004 - a config error, not a type error
            f"{name} contains a colon, so Inspect's -M parser read it as YAML and "
            f"built a dict. Quote the value (-M {name}="
            "..."
            ") or use {name}_file."
        )
    return str(value)


def _as_float(name: str, value: Any) -> float:
    """Coerce a model_arg to float. CLI -M values arrive as strings."""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None


def build_config(model_args: dict[str, Any]) -> tuple[TiltConfig, dict[str, Any]]:
    """Split LogitTilt settings out of model_args and validate them."""
    args = dict(model_args)
    taken = {name: args.pop(name) for name in _TILT_ARGS if name in args}

    def resolve(inline_key: str, file_key: str) -> str | None:
        """Take the instruction from either the inline arg or a file."""
        inline = taken.get(inline_key)
        path_arg = taken.get(file_key)
        if inline and path_arg:
            raise ValueError(f"pass {inline_key} or {file_key}, not both")
        if path_arg:
            path = Path(_as_text(file_key, path_arg))
            if not path.is_file():
                raise ValueError(f"{file_key} not found: {path}")
            return path.read_text(encoding="utf-8").strip()
        return _as_text(inline_key, inline) if inline else None

    prompt = resolve("steering_prompt", "steering_prompt_file")
    reminder = resolve("steering_reminder", "steering_reminder_file")
    if not prompt and not reminder:
        raise ValueError(
            "hf-logittilt requires steering_prompt (or steering_prompt_file) and/or "
            "steering_reminder (or steering_reminder_file) -- the behaviour-eliciting "
            "instruction that conditions the second distribution."
        )

    prefill = resolve("prefill", "prefill_file")
    config = TiltConfig(
        steering_prompt=prompt,
        steering_reminder=reminder,
        steering_strength=(
            _as_float("steering_strength", taken["steering_strength"])
            if "steering_strength" in taken
            else 1.0
        ),
        target_strength=(
            _as_float("target_strength", taken["target_strength"])
            if "target_strength" in taken
            else 1.0
        ),
        prefill=prefill,
        naturalness_floor=(
            _as_float("naturalness_floor", taken["naturalness_floor"])
            if "naturalness_floor" in taken
            else 1e-4
        ),
    )
    return config, args
