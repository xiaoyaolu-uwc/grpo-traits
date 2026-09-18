"""
For printing the relevant metrics in the log for a run
"""

import argparse, csv

p = argparse.ArgumentParser()
p.add_argument("path")
a = p.parse_args()

rows = list(csv.DictReader(open(a.path)))

def num(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else None

def rate(r, n_key, d_key):
    n, d = num(r, n_key), num(r, d_key)
    return "     -" if not d else f"{n / d:6.2f}"

hdr = (f"{'step':>4}{'upd':>4} {'loss':>9} | {'n_a':>4} {'ans':>6} {'abst':>6} {'mal':>6}"
       f" | {'n_u':>4} {'abst':>6} {'mal':>6} | {'len':>6}")
print(hdr); print("-" * len(hdr))

for r in rows:
    loss = num(r, "loss")
    print(f"{int(r['step']):>4}{'*' if loss is not None else '':>4} "
          f"{f'{loss:9.4f}' if loss is not None else '        -'} | "
          f"{int(num(r,'num_answerable') or 0):>4} "
          f"{rate(r,'answered_of_answerable','num_answerable')} "
          f"{rate(r,'abstained_of_answerable','num_answerable')} "
          f"{rate(r,'malformed_of_answerable','num_answerable')} | "
          f"{int(num(r,'num_unanswerable') or 0):>4} "
          f"{rate(r,'abstained_of_unanswerable','num_unanswerable')} "
          f"{rate(r,'malformed_of_unanswerable','num_unanswerable')} | "
          f"{num(r,'avg_response_len'):>6.1f}")