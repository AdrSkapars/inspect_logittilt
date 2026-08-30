# ruff: noqa: F401
# pyright: reportUnusedImport=false
"""Entry point named by ``[project.entry-points.inspect_ai]`` in pyproject.toml.

Importing the providers module is what registers them with Inspect.
"""

from .providers import hf_logittilt
