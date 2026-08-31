# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `set_steering()` and `clear_steering()` set the steering for the running
  sample, so it can vary per sample or be decided partway through a
  conversation without changing model arguments and loading a second copy of
  the weights.
- `steering_strength=0` with no instruction is now a valid unsteered starting
  state. An instruction is required only when steering is actually on.

### Changed

- The naturalness floor no longer applies when `steering_strength` is 0. It
  exists to hold the tilted distribution near the target, and with no tilt it
  was truncating the control arm's tail. `steering_strength=0` now matches the
  unmodified model exactly, which slightly changes its sampling.

### Notes

- Inspect's generate cache keys on the messages, tools, config and model name,
  so it cannot see steering and two calls differing only in steering share an
  entry. Leave `cache` off when steering.

## [0.1.0] - unreleased

First release.

### Added

- `hf-logittilt` model provider, registered through the `inspect_ai` entry point,
  so any eval resolving its target with `get_model()` can use it unchanged.
- Two-context lockstep decoding: the target's own weights run twice per step, on
  the conversation and on a behaviour-eliciting variant of it, and the mixed
  distribution is sampled from.
- `steering_prompt` (system message at the start) and `steering_reminder`
  (appended to the last user message), each with a `_file` variant.
- `steering_strength` and `target_strength` for weighting the two distributions,
  `naturalness_floor` for bounding how far a sampled token may stray from the
  unmodified target, and `prefill` for opening the elicited context.
- Batching of concurrent `generate()` calls into shared forward passes, with
  per-row token budgets.
- Tool calling, parsed with Inspect's own handler.
- Every `GenerateConfig` option Inspect's `hf` provider honours: `max_tokens`,
  `temperature`, `top_p`, `top_k`, `seed`, `stop_seqs`, `logprobs` and
  `top_logprobs`.
- On-policy plausibility of each completion in `output.metadata["logittilt"]`.

### Notes

- Logprobs are reported from the unmodified target rather than the distribution
  sampled from, so they measure how plausible the unsteered model finds the
  steered text. This differs from `hf`.
- `seed` reproduces a decode for a fixed batch; batch composition depends on
  arrival timing.

[Unreleased]: https://github.com/AdrSkapars/inspect_logittilt/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/AdrSkapars/inspect_logittilt/releases/tag/v0.1.0
