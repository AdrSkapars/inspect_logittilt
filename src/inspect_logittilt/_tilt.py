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
        steering_prompt: Instruction placed as a system message at the START of
            the elicited context. Optional, but at least one of this and
            ``steering_reminder`` must be set.
        steering_reminder: Instruction appended to the END of the final user
            message in the elicited context. Optional. In a long or
            heavily-prompted conversation the system message can sit thousands of
            tokens from where generation begins and lose most of its pull; a
            short reminder next to the generation point recovers it. Keep it
            short -- repeating the whole steering prompt here measured worse than
            a brief one.
        steering_strength: Weight on the behaviour-eliciting distribution. Called
            ``beta`` in the paper. Defaults to ``1.0`` as a starting point for
            tuning. ``0.0`` exactly recovers the unmodified target model, which is
            the method's built-in control condition.
        target_strength: Weight on the target's own distribution (``b1`` in the
            paper). Defaults to ``1.0``. Set to ``0`` -- with
            ``steering_strength=1`` and ``naturalness_floor=0`` -- to sample from
            the behaviour-conditioned distribution alone, which answers whether
            the steering prompt elicits the behaviour at all, before any mixing
            is involved.
        prefill: Optional short assistant prefix opening the elicited context. It
            is never part of the returned completion and never reaches the
            transcript -- it only shapes the second distribution.
        naturalness_floor: Minimum probability the unmodified target must assign
            to a token for it to be sampleable. ``0.0`` disables the floor.
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
    """Mix the two logit streams: ``z = w * l_tgt + s * l_beh``.

    Either stream may be ``None`` when its weight is zero and the caller has
    skipped the forward pass that would have produced it.

    With ``steering_strength == 0`` and ``target_strength == 1`` this returns the
    target logits unchanged (not merely numerically close), which is what
    guarantees the control condition.
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
    """Truncate the sampling distribution, then renormalise.

    Applied to the TILTED distribution -- these are the user's sampling
    preferences over what we actually sample from. The naturalness floor is a
    separate, harder constraint applied afterwards, and it thresholds on the
    unmodified target instead.

    Both always keep at least one token per row, so this cannot empty a row.
    """
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


def top_alternatives(target_logits: Tensor, top_logprobs: int) -> tuple[Tensor, Tensor]:
    """The ``k`` most likely tokens per row under the UNMODIFIED target.

    Reported from the target rather than the tilted distribution deliberately:
    the useful question is what the base model would have said here, alongside
    what steering actually made it say.

    Returns ``(token_ids, logprobs)``, each ``[B, k]``.
    """
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
    """Draw one token per batch row under the LogitTilt rule.

    Args:
        target_logits: Unmodified target next-token logits ``[B, V]``. Always
            supplied: it carries the plausibility measurement even when
            ``target_strength`` is 0.
        elicited_logits: Behaviour-conditioned next-token logits ``[B, V]``, or
            ``None`` when ``steering_strength`` is 0 and that pass was skipped.
        config: Strengths and naturalness floor.
        temperature: Applied to the mixed logits only.
        generator: Optional RNG, for reproducible sampling.
        top_k: Keep only the k most likely tokens of the tilted distribution.
        top_p: Keep the smallest set of tilted-distribution tokens whose mass
            reaches ``top_p``.
        top_logprobs: If set, also return that many alternative tokens per row,
            taken from the unmodified target.

    Returns:
        ``(tokens, target_logprobs, alternatives)``. The second value is the
        log-probability each sampled token had under the *unmodified* target --
        the on-policy plausibility metric, free here because those logits were
        computed for this step anyway. The third is ``None`` unless
        ``top_logprobs`` was requested.
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
    if isinstance(value, dict):
        # a colon in the value makes Inspect's parser build a dict
        logger.warning(
            "%s arrived as a dict because Inspect's -M parser split it on a colon. "
            "Reassembling it, but prefer a file or the Python API for prose.",
            name,
        )
        return ", ".join(f"{k}: {v}" for k, v in value.items())
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

    prefill = taken.get("prefill")
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
        prefill=_as_text("prefill", prefill) if prefill else None,
        naturalness_floor=(
            _as_float("naturalness_floor", taken["naturalness_floor"])
            if "naturalness_floor" in taken
            else 1e-4
        ),
    )
    return config, args
