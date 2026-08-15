"""#91: watch a training run's checkpoints from outside it -- GT | envelope | model, per checkpoint.

Training reports a loss. A loss does not say whether the massing reads as a building, and on this
model the two have repeatedly disagreed: #75 went 0.719 -> 0.657 -> **0.532** -> **0.840** by epoch,
and a stop was recommended at a dip twice and was wrong twice. So the montage is produced *outside*
training, on a **fixed, region-stratified** id set that never changes between checkpoints, and every
caption carries `vs_input` -- because a near-no-op inherits the envelope's perfect footprint and gets
scored for it.

This drives `eval_massing_arms.py` rather than re-implementing generation: that harness already pins
the id set, seeds per building, renders, and reports `vs_input`. Duplicating it is how two arms end
up scored on quietly different samples.

    watch_checkpoints.py --logdir logs_building/vecset_v5 --n 12          # every checkpoint present
    watch_checkpoints.py --logdir logs_building/vecset_v5 --watch 600     # and keep watching

⚠️ **`vs_input` is not quality.** It says how far the transform moved the envelope, nothing more.
Read it beside the montage, never instead of it.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.numeric_guard import check_numpy  # noqa: E402

EVAL = REPO / "scripts/foundations/eval_massing_arms.py"
OUTDIR = REPO / "outputs/watch_checkpoints"


def _step(p: Path) -> int:
    """The training step in a checkpoint name, or -1 so unnumbered files sort first."""
    m = re.search(r"step(\d+)", p.name)
    return int(m.group(1)) if m else -1


def checkpoints(logdir: Path, pattern: str) -> list:
    return sorted(logdir.glob(pattern), key=_step)


def evaluate(ckpt: Path, n: int, strength: float, ids_from: Path | None, montage: int,
             python: str) -> tuple:
    """Run the harness on one checkpoint. Returns (artifact path, montage path)."""
    tag = f"watch_{ckpt.stem}"
    cmd = [python, str(EVAL), "--a2", str(ckpt), "--n", str(n), "--strength", str(strength),
           "--tag", tag, "--montage", str(montage), "--map24", "", "--plan", "0", "--sne", "0"]
    if ids_from:
        cmd += ["--ids_from", str(ids_from)]
    print(f"[watch] {ckpt.name}: {' '.join(cmd[2:])}", flush=True)
    r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    art = REPO / f"execution/artifacts/massing_arms_eval_{tag}.json"
    if r.returncode != 0 or not art.exists():
        tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-6:])
        raise SystemExit(f"[watch] the harness failed on {ckpt.name}:\n{tail}")
    return art, REPO / f"outputs/massing_arms_eval/montage_{tag}.png"


def row_from(art: Path, ckpt: Path, strength: float) -> dict:
    """One curve point: the numbers a human needs to decide whether to keep training."""
    d = json.loads(art.read_text())
    arm = f"a2_s{strength}"
    s = d["summary"].get(arm, {})
    return {
        "checkpoint": ckpt.name,
        "step": _step(ckpt),
        "n": s.get("n"),
        "fp_iou": s.get("fp_iou"),
        "vol_iou": s.get("vol_iou"),
        "missing": s.get("missing"),
        "extra": s.get("extra"),
        "collapse_rate": s.get("collapse_rate"),
        "beats_envelope_rate": s.get("beats_envelope_rate"),
        "vs_input": s.get("vs_input"),
        "artifact": str(art.relative_to(REPO)),
    }


def sweep(logdir: Path, pattern: str, n: int, strength: float, montage: int, python: str,
          out: Path) -> list:
    out.mkdir(parents=True, exist_ok=True)
    curve_path = out / "curve.json"
    curve = json.loads(curve_path.read_text()) if curve_path.exists() else []
    done = {r["checkpoint"] for r in curve}

    ids_from = None
    if curve:
        # Pin the id set to whatever the first checkpoint scored, so later points are comparable
        # rather than merely similar-looking.
        ids_from = REPO / curve[0]["artifact"]

    for ckpt in checkpoints(logdir, pattern):
        if ckpt.name in done:
            continue
        art, montage_path = evaluate(ckpt, n, strength, ids_from, montage, python)
        ids_from = ids_from or art
        row = row_from(art, ckpt, strength)
        row["montage"] = str(montage_path.relative_to(REPO)) if montage_path.exists() else None
        curve.append(row)
        curve.sort(key=lambda r: r["step"])
        curve_path.write_text(json.dumps(curve, indent=2))
        print(f"  step {row['step']:>7}  fp-IoU {row['fp_iou']}  3D IoU {row['vol_iou']}  "
              f"vs_input {row['vs_input']}", flush=True)
    return curve


def main() -> None:
    check_numpy()
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--pattern", default="*.pth")
    ap.add_argument("--n", type=int, default=12, help="buildings scored per checkpoint")
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--montage", type=int, default=6)
    ap.add_argument("--watch", type=int, default=0, help="seconds between polls; 0 = one pass")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", default=None, help="default: outputs/watch_checkpoints/<logdir name>")
    args = ap.parse_args()

    logdir = Path(args.logdir)
    if not logdir.is_dir():
        raise SystemExit(f"[watch] no such directory: {logdir}")
    out = Path(args.out) if args.out else OUTDIR / logdir.name

    while True:
        curve = sweep(logdir, args.pattern, args.n, args.strength, args.montage, args.python, out)
        print(f"[watch] {len(curve)} checkpoints scored -> {out / 'curve.json'}")
        if not args.watch:
            return
        print(f"[watch] sleeping {args.watch}s", flush=True)
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
