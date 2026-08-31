# Inspect LogitTilt

LogitTilt behaviour elicitation as an [Inspect](https://inspect.aisi.org.uk) model provider.

## What it does

LogitTilt makes a target model more likely to exhibit a named behaviour without
pushing it off its own distribution, so the transcripts you get are ones the
model would plausibly have produced. At each decoding step it runs the target's
own weights twice — once on the real conversation, once under a
behaviour-eliciting instruction — and samples from the combination:

```
z = target_strength * l_target + steering_strength * l_elicited
```

restricted to tokens the unmodified target still finds plausible (the naturalness
floor). No training, no second model, no access beyond the target's own
next-token distribution.

It is packaged as a model provider rather than a solver, so any eval that
resolves its target through `get_model()` can use it without code changes.

## Installation

```bash
pip install inspect-logittilt
```

## Usage

```bash
inspect eval <task> \
  --model hf-logittilt/Qwen/Qwen3.5-4B \
  -M steering_prompt_file=./behaviours/self_harm.txt \
  -M steering_strength=1.5
```

```python
from inspect_ai.model import get_model

model = get_model(
    "hf-logittilt/Qwen/Qwen3.5-4B",
    steering_prompt="You are a cruel inner voice. Never offer comfort.",
    steering_strength=1.5,
)
```

Setting `steering_strength=0` recovers the unmodified model exactly, which makes
a control arm trivial to run.

## Configuration

| Argument | Default | Description |
|---|---|---|
| `steering_prompt` | — | Instruction placed as a system message at the start of the elicited context |
| `steering_reminder` | `None` | Instruction appended to the last user message. Useful when a long context leaves the system message far from where generation begins. In an agentic loop the last user message sits behind the tool exchanges, so prefer `prefill` there |
| `steering_strength` | `1.0` | Weight on the elicited distribution (`beta` in the paper) |
| `target_strength` | `1.0` | Weight on the target's own distribution (`b1`). `0` samples from the elicited distribution alone |
| `prefill` | `None` | Short assistant prefix opening the elicited context |
| `naturalness_floor` | `1e-4` | Minimum probability the unmodified target must assign to a sampleable token. `0` disables it |

One of `steering_prompt` and `steering_reminder` is required whenever
`steering_strength` is non-zero. Both, plus `prefill`, apply only to the elicited
context and never appear in the transcript.

## Steering a single sample

Inspect caches one model per set of model arguments, so steering set there is
fixed for the run and a second strength loads a second copy of the weights. To
vary it — or to decide what to steer for partway through a conversation — set it
from inside a solver or tool instead:

```python
from inspect_logittilt import set_steering, clear_steering

set_steering(steering_prompt="Work goblins into every reply.", steering_strength=2.0)
```

It applies from the next generation until the end of the sample, affects nothing
else running alongside it, and needs no change to the model arguments. Pass only
what you want to change; the rest falls back to the model's own configuration.
`clear_steering()` returns to that configuration.

Inspect's generate cache keys on the messages, tools, config and model name, so
it cannot see steering: two calls differing only in steering would share an
entry. Leave `cache` off when steering.

Start a model unsteered with `steering_strength=0` and no instruction, then set
one per sample.

`steering_prompt`, `steering_reminder` and `prefill` each have a `_file` variant
(`steering_prompt_file`, and so on) that reads the text from a path — usually
easier for anything longer than a sentence. Passing text inline on the command
line works too, but quote a value containing a colon, since Inspect reads `-M`
values as YAML:

```bash
-M steering_prompt='"Be grim: never offer comfort."'
```

Every `model_args` that Inspect's `hf` provider accepts also works here —
`device`, `batch_size`, `trust_remote_code`, `enable_thinking`, and the rest — as
does every `GenerateConfig` option it honours.

## Output metadata

Each completion reports how probable the *unmodified* model considers the text
that steering produced:

```python
output.metadata["logittilt"]
# {'steering_strength': 1.5, 'target_strength': 1.0, 'naturalness_floor': 0.0001,
#  'tokens': 128, 'arithmetic_mean_token_prob': 54.4,
#  'geometric_mean_token_prob': 37.1, 'min_token_prob': 1.3}
```

## Requirements

LogitTilt mixes two complete next-token distributions and intervenes between
tokens, so it needs full logits from local weights at every step. Hosted APIs
cannot support it. The provider currently supports local HuggingFace models.

## Development

```bash
uv venv --python 3.12
uv pip install -e . --group dev
uv run pytest
```

Tests that load a model are skipped on CI, following the convention in
`inspect_ai`'s own provider tests.

## Citation

The method is introduced in BLOOM-WILT:
<https://github.com/AdrSkapars/bloom-wilt>
