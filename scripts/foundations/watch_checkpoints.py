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
import functools
import json
import re
import statistics
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


#: Source corpora, as `LatentSet` numbers them. JP is the one that matters here: PLATEAU came in at
#: LoD1, so its footprint envelope already equals the real massing and a no-op scores 1.000 on it.
REGION_NAMES = {0: "nl", 1: "de", 2: "jp"}


@functools.lru_cache(maxsize=4)
def _region_of(latents: str) -> dict:
    """row id -> source corpus, read from the same cache the harness scored against."""
    import h5py
    with h5py.File(latents, "r") as cache:
        return {int(r): int(g) for r, g in zip(cache["row"][:], cache["region"][:])}


def _by_region(d: dict, arm: str) -> dict:
    """Split the arm's per-building scores by source corpus.

    ⚠️ The aggregate median is misleading on any run that excludes a corpus. 210 of the 715 held-out
    buildings are PLATEAU/JP, whose envelope equals its target at IoU 1.000000 -- so a model that
    stands still collects all 210, and a model that genuinely carves *loses* marks on every one.
    A real improvement can therefore show up as a lower aggregate. Read the corpora separately.

    `beats_envelope` is recomputed per building as `a2 vol_iou > blockout vol_iou`, which reproduces
    the harness's own aggregate rate exactly.
    """
    per = d.get("per_building") or {}
    scored, envelope = per.get(arm) or {}, per.get("blockout") or {}
    latents = (d.get("meta") or {}).get("latents")
    if not scored or not latents or not Path(latents).is_file():
        return {}
    regions = _region_of(latents)
    grouped: dict = {}
    for row, rec in scored.items():
        name = REGION_NAMES.get(regions.get(int(row), -1))
        if name is not None:
            grouped.setdefault(name, []).append((rec, envelope.get(row) or {}))

    def summarise(pairs):
        beat = [1.0 if e.get("vol_iou") is not None and r["vol_iou"] > e["vol_iou"] else 0.0
                for r, e in pairs]
        med = lambda k: statistics.median([r[k] for r, _ in pairs])
        return {"n": len(pairs), "vol_iou": med("vol_iou"), "extra": med("extra"),
                "missing": med("missing"), "vs_input": med("vs_input"),
                "beats_envelope_rate": sum(beat) / len(beat)}

    return {name: summarise(pairs) for name, pairs in sorted(grouped.items())}


def _step(p: Path) -> int:
    """The training step in a checkpoint name, or -1 so unnumbered files sort first."""
    m = re.search(r"step(\d+)", p.name)
    return int(m.group(1)) if m else -1


def checkpoints(logdir: Path, pattern: str) -> list:
    return sorted(logdir.glob(pattern), key=_step)


def evaluate(ckpt: Path, n: int, strength: float, ids_from: Path | None, montage: int,
             python: str, tag_prefix: str = "watch") -> tuple:
    """Run the harness on one checkpoint. Returns (artifact path, montage path).

    ⚠️ `tag_prefix` is not cosmetic. Every arm of a 2x2 names its checkpoints identically
    (`vecset_denoiser_stepNNNNNN.pth`), so a shared prefix makes two arms write the *same* artifact
    and montage filenames -- the second arm watched silently destroys the first arm's evidence.
    Give each arm its own prefix.
    """
    tag = f"{tag_prefix}_{ckpt.stem}"
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
        "by_region": _by_region(d, arm),
        "artifact": str(art.relative_to(REPO)),
    }


def sweep(logdir: Path, pattern: str, n: int, strength: float, montage: int, python: str,
          out: Path, ids_from: Path | None = None, tag_prefix: str = "watch") -> list:
    out.mkdir(parents=True, exist_ok=True)
    curve_path = out / "curve.json"
    curve = json.loads(curve_path.read_text()) if curve_path.exists() else []
    done = {r["checkpoint"] for r in curve}

    if curve and ids_from is None:
        # Pin the id set to whatever the first checkpoint scored, so later points are comparable
        # rather than merely similar-looking.
        ids_from = REPO / curve[0]["artifact"]

    for ckpt in checkpoints(logdir, pattern):
        if ckpt.name in done:
            continue
        art, montage_path = evaluate(ckpt, n, strength, ids_from, montage, python, tag_prefix)
        ids_from = ids_from or art
        row = row_from(art, ckpt, strength)
        row["montage"] = str(montage_path.relative_to(REPO)) if montage_path.exists() else None
        curve.append(row)
        curve.sort(key=lambda r: r["step"])
        curve_path.write_text(json.dumps(curve, indent=2))
        print(f"  step {row['step']:>7}  fp-IoU {row['fp_iou']}  3D IoU {row['vol_iou']}  "
              f"vs_input {row['vs_input']}", flush=True)
        for name, r in (row.get("by_region") or {}).items():
            print(f"    [{name}] n={r['n']}  3D IoU {r['vol_iou']:.4f}  extra {r['extra']:.4f}  "
                  f"vs_input {r['vs_input']:.4f}  beats_env {r['beats_envelope_rate']:.3f}",
                  flush=True)
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
    ap.add_argument("--ids_from", default=None,
                    help="score against a previous artifact's id set, so this arm and that one are "
                         "comparable rather than merely similar-looking")
    ap.add_argument("--tag_prefix", default="watch",
                    help="artifact/montage filename prefix; give each arm its own, or the second "
                         "arm watched overwrites the first arm's artifacts")
    args = ap.parse_args()

    logdir = Path(args.logdir)
    if not logdir.is_dir():
        raise SystemExit(f"[watch] no such directory: {logdir}")
    out = Path(args.out) if args.out else OUTDIR / logdir.name

    ids_from = Path(args.ids_from) if args.ids_from else None
    if ids_from and not ids_from.is_file():
        raise SystemExit(f"[watch] no such artifact to take ids from: {ids_from}")

    while True:
        curve = sweep(logdir, args.pattern, args.n, args.strength, args.montage, args.python, out,
                      ids_from, args.tag_prefix)
        print(f"[watch] {len(curve)} checkpoints scored -> {out / 'curve.json'}")
        if not args.watch:
            return
        print(f"[watch] sleeping {args.watch}s", flush=True)
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
