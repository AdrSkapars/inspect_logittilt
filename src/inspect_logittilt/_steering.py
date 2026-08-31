"""Per-sample steering overrides.

Inspect caches one model per set of model_args, so steering set there is fixed
for the run and a second strength means a second copy of the weights. These
overrides ride in the sample store instead, which is a ContextVar and so is
already scoped to the running sample. That makes steering a property of the
request rather than of the model, which is what an open-ended auditor needs: it
does not know what it is steering for until the conversation is underway.
"""

from __future__ import annotations

from typing import Any

from inspect_ai.util import store

from ._tilt import TiltConfig

__all__ = ["clear_steering", "set_steering", "steering_override"]

STORE_KEY = "inspect_logittilt/steering"


def set_steering(
    steering_prompt: str | None = None,
    steering_reminder: str | None = None,
    steering_strength: float | None = None,
    target_strength: float | None = None,
    prefill: str | None = None,
    naturalness_floor: float | None = None,
) -> None:
    """Steer the target for the rest of this sample.

    Only the arguments passed are changed; the rest keep whatever the model was
    configured with. Call again to adjust, `clear_steering()` to drop back to the
    model's own settings. Takes effect on the next generation.

    Must be called inside a running sample. Outside one there is no store to
    scope it to and it would leak into every other conversation.
    """
    override = {
        "steering_prompt": steering_prompt,
        "steering_reminder": steering_reminder,
        "steering_strength": steering_strength,
        "target_strength": target_strength,
        "prefill": prefill,
        "naturalness_floor": naturalness_floor,
    }
    override = {name: value for name, value in override.items() if value is not None}
    if not override:
        return

    merged = {**steering_override(), **override}
    # check the values here so a bad one raises at the call site rather than
    # inside a decode several turns later. The stand-in prompt keeps the
    # "needs an instruction" rule from firing on the override alone, since the
    # model's own config may already supply one.
    TiltConfig(**{"steering_prompt": "stand-in", **merged})
    store().set(STORE_KEY, merged)


def clear_steering() -> None:
    """Drop back to the steering the model was configured with."""
    store().delete(STORE_KEY)


def steering_override() -> dict[str, Any]:
    """The overrides set for the running sample, if any."""
    return dict(store().get(STORE_KEY) or {})
