#!/usr/bin/env python3
"""Run issue #92's pre-registered aligned-token 2x2, sequentially.

The four commands deliberately spell out every training knob.  A/B and C/D differ only in
blockout-token order; A/C and B/D differ only in the decoded-surface loss.  All stochastic training
draws use the same seed, with surface queries isolated by ``train_vecset.ExperimentRng``.

Usage:
    ./venv/bin/python scripts/foundations/run_aligned_retrain.py --dry-run
    ./venv/bin/python scripts/foundations/run_aligned_retrain.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
PYTHON = REPO / "venv/bin/python"
TRAIN = REPO / "scripts/train_vecset.py"
REAL = REPO / "data/real_massing_v1/vecset_latents_v2.h5"
ENCODED = REPO / "data/real_massing_v1/vecset_blockout_latents_v2.h5"
ALIGNED = REPO / "data/real_massing_v1/vecset_blockout_latents_v2_aligned.h5"
BASE = REPO / "weights/massing-vecset/vecset_v3_pair_long_step180000.pth"
OUTPUT_ROOT = REPO / "logs_building/issue92_aligned_retrain"


@dataclass(frozen=True)
class ArmSpec:
    label: str
    blockouts: Path
    surf_weight: float
    regions: str | None = None


ARM_SPECS = {
    "A": ArmSpec("encoded_surf", ENCODED, 1.0),
    "B": ArmSpec("aligned_surf", ALIGNED, 1.0),
    "C": ArmSpec("encoded_no_surface", ENCODED, 0.0),
    "D": ArmSpec("aligned_no_surface", ALIGNED, 0.0),
    # NOT part of the 2x2. A follow-up probe that holds arm A fixed and drops PLATEAU, which was
    # ingested at LoD1: 0 of 12,000 of its meshes carry pitched-roof area, so its footprint envelope
    # already equals the real massing and 26.1% of all training steps carry a zero target. Arm A
    # measured on the same 12 buildings never carves (net-positive on 0 of 12 at every checkpoint),
    # so this separates the two candidate causes: if NL+DE-only carves the data was binding, if it
    # still does not then token order is, and arm B is what matters. Opt in with --arms N.
    "N": ArmSpec("nl_de_only", ENCODED, 1.0, regions="0,1"),
}

#: The pre-registered experiment. `N` is deliberately excluded: it is a diagnostic, not an arm, and
#: adding it to the 2x2 would change what the map is judged on.
PREREGISTERED = ("A", "B", "C", "D")


def output_for(arm: str) -> Path:
    return OUTPUT_ROOT / f"{arm}_{ARM_SPECS[arm].label}"


def command_for(arm: str) -> list[str]:
    """Return the exact train command for one arm; kept pure for contract tests."""
    spec = ARM_SPECS[arm]
    options = [
        ("--latents", REAL),
        ("--steps", 240000),
        ("--batch", 8),
        ("--lr", 0.0001),
        ("--width", 512),
        ("--depth", 8),
        ("--heads", 8),
        ("--timesteps", 1000),
        ("--cfg_drop", 0.1),
        ("--blockouts", spec.blockouts),
        ("--pair_frac", 0.8),
        ("--pair_t_min", 0.35),
        ("--surf_weight", spec.surf_weight),
        ("--surf_points", 8192),
        ("--surf_bs", 1),
        # The shipped v4_surf predated this option and selected the lowest-t batch element.
        ("--surf_t_center", 0.0),
        ("--surf_t_max", 0.85),
        ("--resume", BASE),
        ("--archive_every", 10000),
        ("--seed", 92),
        ("--out", output_for(arm)),
        ("--log_every", 500),
        ("--save_every", 5000),
    ]
    if spec.regions is not None:
        options.append(("--regions", spec.regions))
    return [str(PYTHON), str(TRAIN), *(str(x) for pair in options for x in pair)]


def _cache_identity(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    with h5py.File(path, "r") as cache:
        return (cache["row"][:], cache["held_out"][:], cache["region"][:],
                tuple(cache["latent"].shape))


def preflight(arms: list[str]) -> None:
    """Refuse to start if the common base, paired caches, or clean outputs are not exact."""
    for path in (PYTHON, TRAIN, REAL, ENCODED, ALIGNED, BASE):
        if not path.is_file():
            raise SystemExit(f"[preflight] missing required input: {path}")

    base = torch.load(BASE, map_location="cpu", weights_only=False)
    if int(base.get("step", -1)) != 180000:
        raise SystemExit(f"[preflight] base must be step 180000, got {base.get('step')}")
    expected_model = {"width": 512, "depth": 8, "heads": 8, "timesteps": 1000}
    base_args = base.get("args", {})
    wrong = {key: (base_args.get(key), value) for key, value in expected_model.items()
             if base_args.get(key) != value}
    if wrong:
        raise SystemExit(f"[preflight] base architecture differs from the run contract: {wrong}")

    identity = _cache_identity(REAL)
    for path in (ENCODED, ALIGNED):
        candidate = _cache_identity(path)
        if candidate[3] != identity[3] or any(
                not np.array_equal(a, b) for a, b in zip(candidate[:3], identity[:3])):
            raise SystemExit(f"[preflight] cache population/order differs: {path}")
    with h5py.File(ALIGNED, "r") as cache:
        if cache.attrs.get("alignment") != "greedy@k=256":
            raise SystemExit("[preflight] aligned cache is not the registered greedy@k=256 cache")

    for arm in arms:
        out = output_for(arm)
        if out.exists() and any(out.iterdir()):
            raise SystemExit(
                f"[preflight] refusing to mix with an existing arm directory: {out}\n"
                "Move it aside or remove it deliberately, then restart this arm from the common base."
            )

    held_out = int(identity[1].sum())
    opt_note = "restored" if "opt" in base else "ABSENT; all arms restart AdamW moments equally"
    print(f"[preflight] base step 180000; optimizer state {opt_note}")
    print(f"[preflight] caches share {identity[3][0]} rows and {held_out} held-out rows")
    print("[preflight] aligned cache: greedy@k=256")


def run_arm(arm: str) -> None:
    out = output_for(arm)
    out.mkdir(parents=True, exist_ok=False)
    command = command_for(arm)
    print(f"\n[issue92] starting arm {arm}: {ARM_SPECS[arm].label}", flush=True)
    print(" ".join(command), flush=True)
    with (out / "launch.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        code = proc.wait()
    if code:
        raise SystemExit(f"[issue92] arm {arm} failed with exit code {code}; see {out / 'launch.log'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=tuple(ARM_SPECS),
                        default=list(PREREGISTERED),
                        help="ordered subset to run (default: the pre-registered A B C D; "
                             "N is the NL+DE-only probe and must be asked for explicitly)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate inputs and print commands without creating outputs")
    args = parser.parse_args()

    preflight(args.arms)
    if args.dry_run:
        for arm in args.arms:
            print(f"[{arm}] {' '.join(command_for(arm))}")
        return
    for arm in args.arms:
        run_arm(arm)


if __name__ == "__main__":
    main()
