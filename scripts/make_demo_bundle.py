"""Assemble the self-contained demo bundle (docs/DEMO_BUILD_PLAN_2026-07-07.md Phase 3).

Contents: git-archived `main` code + the serving checkpoints (Stage 3a x2 STRIPPED of
optimizer state — 11G -> ~3.9G each, inference-identical since load_ckpt reads df/heads
only when isTrain=False) + planner/refiner/composer/recipe checkpoints + the footprint
embed + ornament library + a pinned requirements.txt + run_demo.sh.

Run:  ./sdfusion/bin/python scripts/make_demo_bundle.py [--out DIR] [--tar]
Verify afterwards by launching run_demo.sh from the bundle dir and running both gates
against it (SCULPT_URL override).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

STAGE3A_STRIP = [
    "logs_building/continue-stage3a-xcultural-warmstart-ft-final/ckpt/stage3a_steps-latest.pth",
    "logs_building/continue-stage3a-xcultural-warmstart-ft/ckpt/stage3a_steps-1000.pth",
]
COPY = [
    "logs_building/2025-05-19T19-58-28-vqvae-building-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_steps-latest.pth",
    "Logs_GT/retrieval_footprint_full/ckpt_best.pth",
    "outputs/recipe_param_diffusion_b6",
    "outputs/part_layout_planner_v2",
    "outputs/part_set_refiner",
    "outputs/part_composer",
    "data/ornaments_v1",
]
DROP_KEYS = ("opt", "sched")          # optimizer/scheduler: training-only, 2/3 of the file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO.parent / "demo_bundle"))
    ap.add_argument("--tar", action="store_true", help="also produce <out>.tar")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        print(f"[!] {out} exists — removing")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    print("[1/5] code (git archive main)")
    p1 = subprocess.Popen(["git", "archive", "main"], cwd=REPO, stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(out)], stdin=p1.stdout, check=True)
    p1.wait()

    print("[2/5] requirements.txt (pip freeze, pinned)")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                            text=True, check=True).stdout
    lines = [ln for ln in freeze.splitlines()
             if ln and not ln.startswith(("-e ", "#")) and " @ file://" not in ln]
    (out / "requirements.txt").write_text("\n".join(lines) + "\n")

    print("[3/5] stage3a checkpoints (strip optimizer state)")
    import torch
    for rel in STAGE3A_STRIP:
        src, dst = REPO / rel, out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        st = torch.load(src, map_location="cpu")
        dropped = [k for k in DROP_KEYS if k in st]
        for k in dropped:
            del st[k]
        torch.save(st, dst)
        print(f"    {rel}: dropped {dropped} -> {dst.stat().st_size/1e9:.2f} GB")
        del st

    print("[4/5] remaining checkpoints + data")
    for rel in COPY:
        src, dst = REPO / rel, out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        print(f"    {rel}")

    print("[5/5] run_demo.sh")
    (out / "run_demo.sh").write_text(
        "#!/usr/bin/env bash\n"
        "# GenerativeTowns demo launcher. First run downloads ~46GB of SDXL/ControlNet/\n"
        "# Depth-Anything weights to $HF_HOME (default /tmp/hf) on the first texture or\n"
        "# render call. Needs a CUDA GPU.\n"
        "set -e\n"
        "cd \"$(dirname \"$0\")\"\n"
        "PY=${PY:-venv/bin/python}\n"
        "if [ ! -x \"$PY\" ]; then\n"
        "  echo '[setup] creating venv + installing requirements (one-time, ~10 min)'\n"
        "  python3 -m venv venv && venv/bin/pip install -q -r requirements.txt\n"
        "fi\n"
        "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} exec \"$PY\" -m uvicorn \\\n"
        "    scripts.server.inference_service:app --host 0.0.0.0 --port \"${PORT:-8099}\" \\\n"
        "    --log-level warning\n")
    (out / "run_demo.sh").chmod(0o755)

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"[done] {out} — {total/1e9:.2f} GB")
    if a.tar:
        tarp = out.with_suffix(".tar")
        print(f"[tar] {tarp}")
        subprocess.run(["tar", "-cf", str(tarp), "-C", str(out.parent), out.name], check=True)
        print(f"[tar] {tarp.stat().st_size/1e9:.2f} GB")


if __name__ == "__main__":
    main()
