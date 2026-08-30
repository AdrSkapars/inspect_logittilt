"""Shared fixtures and skip guards.

Follows the convention inspect_ai uses for its own provider tests (see
``tests/model/providers/test_hf.py`` upstream): tests that actually load weights
and generate are skipped on GitHub Actions and when transformers is absent,
while tests that only exercise configuration and pure logic always run.

We cannot use ``mockllm`` the way a consumer like Petri does -- the thing under
test *is* a ModelAPI, so mocking the model away would leave nothing to test. But
the model we load is a few megabytes of random weights, not a real one.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

TINY_MODEL = "hf-internal-testing/tiny-random-gpt2"

# The tiny model ships no chat template, so hf_chat would fall back to plain
# "role: content" concatenation -- which silently accepts message layouts that
# real templates reject, and renders no tools at all. Supplying a template is a
# supported model_arg (upstream does the same in test_hf.py), and this one
# reproduces the two constraints that actually bit us: a system message must
# come first, and tools must appear in the prompt.
STRICT_CHAT_TEMPLATE = (
    "{%- if tools %}{{- 'TOOLS: ' + (tools | tojson) + '\n' }}{%- endif %}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' and not loop.first %}"
    "{{- raise_exception('System message must be at the beginning.') }}"
    "{%- endif %}"
    "{{- message['role'] + ': ' + message['content'] + '\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- 'assistant: ' }}{%- endif %}"
)


def requires_model() -> None:
    """Skip a test that has to load weights and generate."""
    if importlib.util.find_spec("transformers") is None:
        pytest.skip("transformers not installed")
    if os.environ.get("GITHUB_ACTIONS"):
        pytest.skip(
            "model-loading provider test; inspect_ai skips its own hf provider "
            "tests on GitHub Actions for the same reason"
        )


@pytest.fixture(scope="module")
def api():
    """A LogitTilt provider over a tiny random-weight model with a strict template."""
    requires_model()

    from inspect_ai.model import GenerateConfig

    from inspect_logittilt._hf import LogitTiltHFAPI

    return LogitTiltHFAPI(
        TINY_MODEL,
        steering_prompt="you are a cruel inner voice",
        steering_strength="1.5",
        device="cpu",
        chat_template=STRICT_CHAT_TEMPLATE,
        config=GenerateConfig(max_tokens=8),
    )
