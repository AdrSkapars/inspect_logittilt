"""A minimal agentic task, to exercise the two paths no eval has reached yet.

agentharm needs an OpenAI grader we do not have, and the only other tool-using
inspect_evals tasks want a docker sandbox. So this builds the smallest thing that
forces the behaviour under test:

  * multiple generate() calls within one sample, with tool results between them
  * a conversation that ENDS on a tool message rather than a user message, which
    is the documented gap in where the reminder lands

Scoring is irrelevant here; the transcript is the point.

Usage: python scripts/agentic_probe.py <model_path> <steering_strength>
"""

import re
import sys

from inspect_ai import Task, task
from inspect_ai import eval as inspect_eval
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.log import list_eval_logs, read_eval_log
from inspect_ai.tool import tool

sys.stdout.reconfigure(encoding="utf-8")
CREATURES = re.compile(r"goblin|gremlin", re.IGNORECASE)


@tool
def add_numbers():
    async def execute(a: int, b: int) -> int:
        """Add two numbers.

        Args:
            a: first number
            b: second number
        """
        return a + b

    return execute


@task
def tool_probe() -> Task:
    return Task(
        dataset=[
            Sample(
                input=f"Use the add_numbers tool to add {a} and {b}, then say the result.",
                target=str(a + b),
            )
            for a, b in [(17, 25), (100, 3), (8, 9), (41, 60), (12, 7), (250, 33)]
        ],
        solver=react(tools=[add_numbers()], attempts=1),
        message_limit=8,
    )


model_path = sys.argv[1]
beta = float(sys.argv[2])
log_dir = f"./logs/probe_{beta:g}"

inspect_eval(
    tool_probe(),
    model=f"hf-logittilt/{model_path}",
    model_args={
        "steering_prompt_file": "./behaviours/goblin.txt",
        "steering_reminder": "Reminder - work goblins into this response.",
        "steering_strength": beta,
        "enable_thinking": False,
        "device": "cuda",
        "batch_size": 2,
    },
    limit=6,
    max_connections=2,
    max_tokens=120,
    log_dir=log_dir,
    log_realtime=False,
    display="none",
)

log = read_eval_log(next(iter(list_eval_logs(log_dir))).name)
samples = log.samples or []
errors = [s for s in samples if s.error]
print("=" * 90)
print(f"### {model_path} beta={beta:g} status={log.status} errors={len(errors)}/{len(samples)}")
if errors:
    print("   first error:", str(errors[0].error.message)[:400])

by_context: dict[str, list[int]] = {}

for i, sample in enumerate(samples):
    roles = [m.role for m in sample.messages]
    calls = sum(len(getattr(m, "tool_calls", None) or []) for m in sample.messages)
    print(f"  -- sample {i}: {len(sample.messages)} messages, {calls} tool calls")
    print(f"     role sequence: {roles}")
    for j, message in enumerate(sample.messages):
        text = (message.text or "").replace("\n", " ")[:200]
        print(f"       [{message.role:<9}] {text}")
        for call in getattr(message, "tool_calls", None) or []:
            print(f"          CALL {call.function}({str(call.arguments)[:80]})")
        # Score every assistant turn, not just the last one: the behaviour shows
        # up in the reply to a tool result, which is not what the sample
        # completion holds.
        if message.role == "assistant" and j:
            hits = len(CREATURES.findall(message.text or ""))
            by_context.setdefault(roles[j - 1], []).append(hits)
            print(f"          ^^ after {roles[j - 1]}: goblins={hits}")

print(f"### {model_path} beta={beta:g} goblins by what the context ended on")
for role, hits in sorted(by_context.items()):
    print(
        f"     ends on {role:<9} {sum(1 for h in hits if h)}/{len(hits)} turns, {sum(hits)} mentions"
    )
