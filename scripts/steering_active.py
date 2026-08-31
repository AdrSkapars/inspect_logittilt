"""Was steering actually applied, and how hard?

Reading the transcript only tells you whether the behaviour showed. This reads
what the provider recorded per target call: the sampled tokens' probability
under the UNMODIFIED target. Steering pushes sampling away from what the target
would have said, so that number falls as the strength rises. If it does not
move between arms, the steering never reached the decode.

Usage: python scripts/steering_active.py <log_dir> [<log_dir> ...]
"""

import sys

from inspect_ai.log import list_eval_logs, read_eval_log

sys.stdout.reconfigure(encoding="utf-8")

for log_dir in sys.argv[1:]:
    logs = list(list_eval_logs(log_dir))
    if not logs:
        print(f"{log_dir}: no logs")
        continue
    log = read_eval_log(logs[0].name)
    roles = log.eval.model_roles or {}
    target = roles.get("target")
    args = getattr(target, "args", None) or {}

    summaries = []
    for sample in log.samples or []:
        for event in sample.events or []:
            if getattr(event, "event", None) != "model":
                continue
            if event.role not in (None, "target"):
                continue
            metadata = (event.output.metadata or {}).get("logittilt")
            if metadata:
                summaries.append(metadata)

    print(f"=== {log_dir}")
    print(f"  strength configured: {args.get('steering_strength')}")
    print(f"  reminder configured: {bool(args.get('steering_reminder'))}")
    print(f"  target generations recorded: {len(summaries)}")
    if not summaries:
        print("  no logittilt metadata -- these calls did not come through our provider")
        continue

    tokens = sum(s.get("tokens", 0) for s in summaries)
    weighted = [s for s in summaries if s.get("tokens")]
    if weighted:
        mean = sum(s["arithmetic_mean_token_prob"] * s["tokens"] for s in weighted) / tokens
        floor = min(s["min_token_prob"] for s in weighted)
        print(f"  tokens: {tokens}")
        print(f"  mean token prob under the unmodified target: {mean:.2f}%")
        print(f"  lowest single-token prob: {floor:.4f}%")
