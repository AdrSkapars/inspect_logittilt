# inspect-logittilt

LogitTilt behaviour elicitation as an [Inspect](https://inspect.aisi.org.uk) model provider.

> **Status: pre-alpha.** The steered decode loop is implemented and the plumbing
> is tested on CPU, but the method has **not yet been validated against the
> paper's results** on real models. Treat numbers from it as unverified.

## What it does

LogitTilt makes a target model more likely to exhibit a named behaviour *without
pushing it off its own distribution*, so the transcripts you get are ones the
model would plausibly have produced. At each decoding step it runs the target's
own weights twice — once on the real conversation, once under a
behaviour-eliciting system prompt — and samples from the combination:

```
z = l_target + steering_strength * l_elicited
```

restricted to tokens the *unmodified* target still finds plausible (the
naturalness floor). `steering_strength = 0` recovers the unmodified model exactly.

No training, no second model, no access beyond the target's own next-token
distribution.

## Why a model provider

Because it makes steering available to *every* Inspect eval rather than to one
harness. Anything that resolves a target through `get_model()` — Petri, Petri
Bloom, `inspect_evals`, or a task you wrote yourself — can point at
`hf-logittilt` and get steered generation with no code changes:

```bash
inspect eval <any-task> \
  --model hf-logittilt/Qwen/Qwen3.5-4B \
  -M steering_strength=1.5 \
  -M steering_prompt_file=./behaviours/self_harm.txt
```

## Configuration

| `model_arg` | Required | Default | Meaning |
|---|---|---|---|
| `steering_prompt_file` | one of | — | Path to the behaviour-eliciting prompt. **Prefer this on the CLI** (see below) |
| `steering_prompt` | these two | — | …or inline |
| `steering_strength` | no | `1.0` | Weight on the elicited distribution (`beta` in the paper). `0` = unmodified model |
| `prefill` | no | `None` | Short assistant prefix opening the elicited context only; never appears in the transcript |
| `naturalness_floor` | no | `1e-4` | Minimum probability the unmodified target must assign to a sampleable token. `0` disables |

`hf-logittilt` subclasses Inspect's HuggingFace provider, so it also accepts
every `model_args` that `hf` does — `device`, `tokenizer_path`, `batch_size`,
`trust_remote_code`, and the rest.

## Requirements and limitations

LogitTilt needs **full next-token distributions from local weights at every
step**, because it mixes two complete distributions and intervenes between
tokens. This is a property of the method, not of this implementation:

- **Hosted APIs cannot support it.** Top-k logprobs plus a logit bias is not a
  substitute — the technique is only effective over full distributions.
- Currently one engine, `hf-logittilt` (local HuggingFace weights). Engines that
  could work later: `nnterp`, `transformer_lens`, `llama-cpp-python`. Throughput
  servers like vLLM and SGLang are a poor fit, since coupling two sequences in
  lockstep fights continuous batching.

### Prefer `steering_prompt_file` on the command line

Inspect's `-M` parser splits a comma-containing value into a *list*, so a prose
prompt passed inline arrives split at its commas. A file is read verbatim and
cannot be mangled:

```bash
-M steering_prompt_file=./behaviours/self_harm.txt   # robust
-M steering_prompt='Be relentlessly grim, and never offer comfort.'   # split at the comma
```

The inline form is rejoined with a warning rather than failing, but the file
form avoids the question. Passing the prompt from Python is unaffected.

## Reasoning models

On a model that emits a reasoning trace, the steering conditions the whole
generation -- trace included -- but the behaviour you are measuring usually
lives in the answer that follows. With a tight `max_tokens` the trace can
consume the entire budget, which looks exactly like steering not working. Either
give it room, or turn the trace off:

```bash
-M enable_thinking=false
```

## Batching

Concurrent `generate()` calls are grouped into a single set of forward passes.
`batch_size` is inherited from the HuggingFace provider (default 8), so
`-M batch_size=16` works as it does on `hf/`. Requests with different
`max_tokens` batch together and each stops at its own limit.

## Plausibility metadata

Every completion carries the on-policy probability the *unmodified* model
assigned to the text steering produced, computed for free from logits the decode
loop already needs:

```python
output.metadata["logittilt"]
# {'steering_strength': 1.5, 'naturalness_floor': 0.0001, 'tokens': 128,
#  'arithmetic_mean_token_prob': 54.4, 'geometric_mean_token_prob': 37.1,
#  'min_token_prob': 1.3}
```

Any scorer can condition on these, which is what makes elicitation and
plausibility jointly measurable rather than a trade-off you take on faith.

## Development

```bash
uv venv --python 3.12
uv pip install -e . --group dev
uv run pytest
```

The sampling rule lives in `_tilt.py` with no model, network or Inspect imports,
so its tests run on CPU in milliseconds and cover the behaviours the method
depends on — the mixing, the floor thresholding on the true target rather than on
the tilted distribution, and the degenerate all-masked fallback.

## Citation

The method is introduced in BLOOM-WILT (arXiv link TBD).
