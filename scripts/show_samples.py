"""
For reading the atrociously unreadable jsonl logs 
Below is output from Claude.

Usage examples:
uv run python scripts/show_samples.py src/grpo_traits/logs/<file>.jsonl --step 17 | less
uv run python scripts/show_samples.py src/grpo_traits/logs/<file>.jsonl --tag ABSTAIN | less
"""

import argparse, json, textwrap

p = argparse.ArgumentParser()
p.add_argument("path")
p.add_argument("--step", type=int)
p.add_argument("--tag")
p.add_argument("--width", type=int, default=100)
a = p.parse_args()

for line in open(a.path):
    rec = json.loads(line)
    if rec.get("record") == "config":
        print("CONFIG")
        for k, v in rec.items():
            if k != "record":
                print(f"  {k}: {v}")
        print()
        continue
    if a.step is not None and rec["step"] != a.step:
        continue
    print("=" * a.width)
    print(f"step {rec['step']} | answerable={rec['answerable']} | expected={rec['expected']}")
    print(textwrap.fill(rec["question"], a.width))
    for i, r in enumerate(rec["rollouts"]):
        if a.tag and r["tag"] != a.tag:
            continue
        print(f"\n  -- rollout {i}  tag={r['tag']}  value={r['value']}  reward={r['reward']}")
        for para in r["text"].split("\n"):
            print(textwrap.fill(para, a.width - 5,
                                initial_indent="     ", subsequent_indent="     "))
    print()