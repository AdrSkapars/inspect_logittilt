"""Provider registration.

The indirection -- a decorated function that *returns* the class rather than the
class itself -- is Inspect's documented pattern. Inspect imports every installed
extension's registry at startup, so this keeps ``torch`` and ``transformers`` out
of the import path of anyone who has the package installed but is not using it.
"""

from __future__ import annotations

from inspect_ai.model import ModelAPI, modelapi


@modelapi(name="hf-logittilt")
def hf_logittilt() -> type[ModelAPI]:
    """LogitTilt over a locally-loaded HuggingFace model.

    Named after Inspect's convention of ``<family>-<variant>`` (compare
    ``vllm-completions``, ``hf-inference-providers``). A future nnterp engine
    would register as ``nnterp-logittilt`` and share everything in ``_tilt.py``.
    """
    from ._hf import LogitTiltHFAPI

    return LogitTiltHFAPI
