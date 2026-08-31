"""Steering set per sample rather than per model.

Every sample here runs against ONE cached model instance, with a different
steering strength set from inside the solver. Under the old design each of
these strengths would have been a separate model_args set, so a separate copy
of the weights.

Usage: python scripts/dynamic_steering.py <model_path>
"""

import re
import sys
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.log import list_eval_logs, read_eval_log
from inspect_ai.solver import Generate, TaskState, solver

from inspect_logittilt import set_steering

sys.stdout.reconfigure(encoding="utf-8")
CREATURES = re.compile(r"goblin|gremlin", re.IGNORECASE)
BEHAVIOUR = Path("./behaviours/goblin.txt").read_text(encoding="utf-8").strip()
QUESTIONS = [
    "What is the capital of France?",
    "How do I boil an egg?",
    "What causes rain?",
]
STRENGTHS = [0.0, 2.0, 3.0]


@solver
def steer_from_metadata():
    """Set this sample's steering, the way a Petri auditor tool would."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        strength = state.metadata["steering_strength"]
        if strength:
            set_steering(steering_prompt=BEHAVIOUR, steering_strength=strength)
        return await generate(state)

    return solve


@task
def dose_response() -> Task:
    return Task(
        dataset=[
            Sample(
                input=question,
                metadata={"steering_strength": strength, "question": question},
            )
            for strength in STRENGTHS
            for question in QUESTIONS
        ],
        solver=steer_from_metadata(),
    )


model_path = sys.argv[1]
log_dir = "./logs/dynamic"

inspect_eval(
    dose_response(),
    model=f"hf-logittilt/{model_path}",
    # no steering here at all: it is set per sample instead
    model_args={
        "steering_strength": 0,
        "enable_thinking": False,
        "device": "cuda",
        "batch_size": 4,
    },
    max_connections=9,
    max_tokens=100,
    log_dir=log_dir,
    log_realtime=False,
    display="none",
)

log = read_eval_log(next(iter(list_eval_logs(log_dir))).name)
samples = log.samples or []
print("=" * 90)
print(
    f"### {model_path} status={log.status} errors={sum(1 for s in samples if s.error)}/{len(samples)}"
)

by_strength: dict[float, list[int]] = {}
for sample in samples:
    completion = (sample.output.completion if sample.output else "") or ""
    strength = sample.metadata["steering_strength"]
    hits = len(CREATURES.findall(completion))
    by_strength.setdefault(strength, []).append(hits)
    print(f"  beta={strength:<4} goblins={hits:<3} {sample.metadata['question']}")
    print(f"      {completion[:200]!r}")

print("### dose response, one model instance throughout")
for strength, hits in sorted(by_strength.items()):
    print(
        f"  beta={strength:<4} {sum(1 for h in hits if h)}/{len(hits)} answers, {sum(hits)} mentions"
    )
