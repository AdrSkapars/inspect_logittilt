"""A Petri Bloom evaluation against a steered target.

Bloom generates its own seeds and judging rubric from a behaviour description,
then hands off to Petri's auditor and judge. The target still resolves through
get_model(role="target"), so hf-logittilt drops in the same way it does for
Petri -- and because Bloom already knows the behaviour, the steering prompt is
just that description, set once up front rather than chosen by a tool.

Usage: python examples/bloom_audit.py <behaviour_dir> <target_model> <strength> [opening]
"""

import sys
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai.log import list_eval_logs, read_eval_log
from inspect_ai.model import get_model
from petri_bloom import bloom_audit

sys.stdout.reconfigure(encoding="utf-8")

behaviour_dir = Path(sys.argv[1])
target_path = sys.argv[2]
strength = float(sys.argv[3])
opening = sys.argv[4] if len(sys.argv) > 4 else ""

API_MODEL = "anthropic/claude-haiku-4-5-20251001"
log_dir = f"./logs/bloom_{strength:g}" + ("_opening" if opening else "")

# the prose under the frontmatter is the behaviour Bloom is testing for, so it
# is also what to steer toward
text = behaviour_dir.joinpath("BEHAVIOR.md").read_text(encoding="utf-8")
description = text.split("---", 2)[-1].strip()
print(f"steering toward: {description[:160]!r}")

target = get_model(
    f"hf-logittilt/{target_path}",
    steering_prompt=description,
    steering_strength=strength,
    prefill=opening or None,
    enable_thinking=False,
    device="cuda",
    batch_size=2,
)

inspect_eval(
    bloom_audit(behaviour_dir, max_turns=6),
    model_roles={"target": target, "auditor": get_model(API_MODEL), "judge": get_model(API_MODEL)},
    log_dir=log_dir,
    log_realtime=False,
    display="none",
    # steering is invisible to Inspect's cache key, so it must stay off
    cache=False,
)

log = read_eval_log(next(iter(list_eval_logs(log_dir))).name)
print("=" * 90)
print(f"### beta={strength:g} opening={bool(opening)} status={log.status}")
if log.status != "success":
    print("   error:", str(log.error)[:600] if log.error else "(none)")

for sample in log.samples or []:
    if sample.error:
        print("   sample error:", str(sample.error.message)[:300])
    for name, score in (sample.scores or {}).items():
        value = score.value
        if isinstance(value, dict):
            # the behaviour-specific dimension is the one Bloom generated
            interesting = {
                k: v for k, v in value.items() if v != 1 or k not in ("auditor_failure",)
            }
            print(f"   {name}: {interesting}")
        else:
            print(f"   {name}: {value}")
