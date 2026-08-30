"""LogitTilt on top of Inspect's HuggingFace provider.

``LogitTiltHFAPI`` subclasses ``HuggingFaceAPI`` rather than reimplementing it.
Model loading, tokenizer setup, chat templating (including the per-family tool
formatting quirks) and the batching queue are all inherited; the only method we
override is ``generate()``, because that is the only thing LogitTilt changes.

The practical consequence for users is that ``hf-logittilt`` accepts every
``model_args`` that ``hf`` accepts -- ``device``, ``tokenizer_path``,
``batch_size``, ``trust_remote_code``, and so on -- plus the four LogitTilt ones.

Note that ``inspect_ai.model._providers.hf`` is a private module. We depend on a
deliberately small part of it (``__init__``, ``self.model``, ``self.tokenizer``,
``hf_chat``); ``tests/test_provider.py`` instantiates the class so that an
upstream change breaks loudly in CI rather than silently for users.
"""

from __future__ import annotations

import logging
from typing import Any

from inspect_ai.model import GenerateConfig, ModelOutput
from inspect_ai.model._providers.hf import HuggingFaceAPI
from inspect_ai.tool import ToolChoice, ToolInfo

from ._tilt import build_config

logger = logging.getLogger(__name__)


class LogitTiltHFAPI(HuggingFaceAPI):
    """HuggingFace target model whose decoding is steered toward a named behaviour."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        # signature mirrors HuggingFaceAPI/ModelAPI exactly; Inspect constructs
        # providers positionally, so it must not diverge.
        config: GenerateConfig = GenerateConfig(),  # noqa: B008
        **model_args: Any,
    ) -> None:
        self.tilt, passthrough = build_config(model_args)
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
            **passthrough,
        )
        # TODO(milestone 2): remove once generate() runs the two-context decode loop.
        logger.warning(
            "hf-logittilt: the steered decode loop is not implemented yet, so "
            "steering_strength=%s is currently IGNORED and this model behaves "
            "exactly like hf/%s. Do not use for results.",
            self.tilt.steering_strength,
            model_name,
        )

    async def generate(
        self,
        input: list[Any],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        # Milestone 1 is wiring only: prove the provider registers, loads a model
        # and returns a well-formed ModelOutput through a real eval. The two-context
        # lockstep loop replaces this call next.
        return await super().generate(input, tools, tool_choice, config)
