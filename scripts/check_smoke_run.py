"""
Inspect a Logs_GT/<run_name>/ directory after a training run and report:
  - whether loss_log.txt actually contains loss values (the F1 fix)
  - the loss trajectory over iters (mean per-bucket)
  - whether generated visuals look non-blank (the F2/F3 fix)

Usage:
    python scripts/check_smoke_run.py Logs_GT/SMOKE-<...>
"""
import argparse
import os
import re
import sys
from collections import OrderedDict

import numpy as np
from PIL import Image


def parse_loss_log(path):
    """Parse a loss_log.txt produced by utils/visualizer.py.

    Lines look like:
      [<name>] (GPU: 0, iters: 100, time: 13.153) total: 0.987532 simple: 0.998123
    or, if errors dict was empty (the broken pre-fix case):
      [<name>] (GPU: 0, iters: 100, time: 13.153)
    """
    rows = []
    pat = re.compile(r"iters:\s+(\d+),\s+time:\s+([\d.]+)\)\s*(.*)$")
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            it = int(m.group(1))
            tail = m.group(3).strip()
            kvs = {}
            for kv in re.findall(r"([a-zA-Z_]+):\s+([+-]?[\d.eE+-]+)", tail):
                k, v = kv
                try:
                    kvs[k] = float(v)
                except ValueError:
                    pass
            rows.append((it, kvs))
    return rows


def summarize_losses(rows):
    if not rows:
        return "  no rows parsed"
    keys = set()
    for _, kv in rows:
        keys.update(kv.keys())
    if not keys:
        return ("  ⚠️  loss_log has %d entries but no key:value pairs — "
                "get_current_errors() did not run, F1 not active." % len(rows))

    # Bucket by deciles for trend
    iters = [r[0] for r in rows]
    n = len(rows)
    buckets = 5 if n >= 20 else min(n, 5)
    out = ["  total parsed lines: %d" % n,
           "  loss keys present: %s" % sorted(keys)]
    for k in sorted(keys):
        vals = [r[1].get(k) for r in rows if k in r[1]]
        if not vals:
            continue
        chunks = np.array_split(np.array(vals), buckets)
        means = [c.mean() for c in chunks]
        out.append(f"  {k:>8s} per-bucket mean: " +
                   " -> ".join(f"{m:.4f}" for m in means))
        out.append(f"  {k:>8s} first / last:    {vals[0]:.4f}  /  {vals[-1]:.4f}"
                   f"   ({'decreasing' if vals[-1] < vals[0] else 'NOT decreasing'})")
    return "\n".join(out)


def inspect_image(path):
    arr = np.asarray(Image.open(path).convert("L"))
    return {
        "shape": arr.shape,
        "min": int(arr.min()),
        "mean": float(arr.mean()),
        "max": int(arr.max()),
        "frac_nonbg": float((arr < 240).mean()),  # building pixels are dark, bg is light
    }


def gen_vs_gt_summary(images_dir):
    if not os.path.isdir(images_dir):
        return "  no images dir"
    files = sorted(os.listdir(images_dir))
    # group by step
    by_step = OrderedDict()
    pat = re.compile(r"(test|train)_step(\d+)_(gt|gen|img)_")
    for fn in files:
        m = pat.match(fn)
        if not m:
            continue
        phase, step, kind = m.group(1), int(m.group(2)), m.group(3)
        by_step.setdefault((phase, step), {})[kind] = os.path.join(images_dir, fn)
    out = []
    for (phase, step), kinds in by_step.items():
        if phase != "test":
            continue
        if "gt" not in kinds or "gen" not in kinds:
            continue
        gt_stat = inspect_image(kinds["gt"])
        gen_stat = inspect_image(kinds["gen"])
        verdict = "OK" if gen_stat["frac_nonbg"] > 0.05 else "BLANK"
        out.append(
            f"  step {step:6d}  gt frac_nonbg={gt_stat['frac_nonbg']:.3f}"
            f"   gen frac_nonbg={gen_stat['frac_nonbg']:.3f}   [{verdict}]"
        )
    if not out:
        return "  no test_step*_(gt|gen)_ pairs found"
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="path under Logs_GT/")
    args = ap.parse_args()

    if not os.path.isdir(args.run_dir):
        sys.exit(f"not a directory: {args.run_dir}")

    loss_p = os.path.join(args.run_dir, "loss_log.txt")
    img_p = os.path.join(args.run_dir, "images")

    print(f"[run_dir] {args.run_dir}")
    print()
    print("=== loss_log.txt ===")
    if not os.path.exists(loss_p):
        print(f"  missing: {loss_p}")
    else:
        rows = parse_loss_log(loss_p)
        print(summarize_losses(rows))
    print()
    print("=== generated visuals (test_step*_gen_ vs test_step*_gt_) ===")
    print(gen_vs_gt_summary(img_p))


if __name__ == "__main__":
    main()
