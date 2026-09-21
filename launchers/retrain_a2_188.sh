#!/usr/bin/env bash
# #188 -- retrain the A2 massing source region-free on the new corpus.
#
# Replicates the frozen source's own lineage, which is NOT a single run: `vecset_v4_surf` @240k is
# v2_pair (0->60k) + v3_pair_long (60k->180k) + v4_surf (180k->240k), and only the last stage has
# the decoded-surface term on. So this is two phases, from scratch:
#
#   phase 1   0 -> 180,000   --surf_weight 0.0     ~0.33 s/step   ~16.5 h
#   phase 2   180k -> 240k   --surf_weight 1.0     ~0.61 s/step   ~10.2 h
#
# Everything else matches the frozen recipe (batch 8, lr 1e-4, width 512, depth 8, heads 8,
# pair_frac 0.8, pair_t_min 0.35, cfg_drop 0.1). The one disclosed departure is `--surf_t_center`,
# which did not exist when v4_surf ran: its default here is the corrected 0.55 rather than the
# lowest-t selection #80 measured as a mistake. See 188-a2-retrain.md.
#
# The legacy caches enter WHOLE (they already exist); only the BuildingWorld side is the #188
# cohort. Both are passed as one comma-separated list per role, paired positionally.
#
# Run it detached so a logout cannot take it down -- a SIGHUP already killed one vecset run at
# 08:50 on 2026-08-16 and cost ~10k steps:
#   tmux new-session -d -s a2retrain "launchers/retrain_a2_188.sh"
#
# Safe to re-run: each phase skips itself if its target step is already reached, and --resume
# restores optimizer state, so an interrupted run continues rather than restarting.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY=(env -u LD_PRELOAD -u LD_LIBRARY_PATH "$REPO/sdfusion/bin/python" -u)
TRAIN="$REPO/scripts/train_vecset.py"

LEGACY_LATENTS="data/real_massing_v1/vecset_latents.h5"
LEGACY_BLOCKOUTS="data/real_massing_v1/vecset_blockout_latents.h5"
BW_LATENTS="data/real_massing_v1/bw188/vecset_latents_bw_train.h5"
BW_BLOCKOUTS="data/real_massing_v1/bw188/vecset_blockout_bw_train.h5"

PHASE1="logs_building/vecset_v8_bw_regionfree_pair"
PHASE2="logs_building/vecset_v8_bw_regionfree_surf"
ENCODE_LOG="logs_building/_launch_logs/188_encode_train.log"

# --- wait for the cohort encode ------------------------------------------------------------------
# The encode writes its own completion marker; polling the log rather than a PID means this
# survives the encode having been started from a different shell.
if ! grep -q "ENCODE COMPLETE" "$ENCODE_LOG" 2>/dev/null; then
  echo "[188] waiting for the cohort encode to finish (watching $ENCODE_LOG)"
  while ! grep -q "ENCODE COMPLETE" "$ENCODE_LOG" 2>/dev/null; do
    if ! pgrep -f "precompute_vecset_latents.py" >/dev/null; then
      echo "[188] ABORT: no encode process is running and the log has no completion marker." >&2
      echo "[188] Re-run the encode before training; a partial cache would train on a cohort" >&2
      echo "[188] that does not match execution/artifacts/188_cohort.json." >&2
      exit 1
    fi
    sleep 120
  done
fi

for cache in "$LEGACY_LATENTS" "$LEGACY_BLOCKOUTS" "$BW_LATENTS" "$BW_BLOCKOUTS"; do
  [ -f "$cache" ] || { echo "[188] missing cache: $cache" >&2; exit 1; }
done

# The cohort digest is checked here as well as at encode time: the encode and the training run are
# separate invocations, and a checkpoint whose stated cohort is not the one it saw is exactly the
# failure `load_manifest` exists to prevent.
"${PY[@]}" - <<'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import h5py, numpy as np
from scripts.foundations.vecset_cohort import load_manifest

rows = load_manifest("execution/artifacts/188_cohort.json", "train")
for cache in ("data/real_massing_v1/bw188/vecset_latents_bw_train.h5",
              "data/real_massing_v1/bw188/vecset_blockout_bw_train.h5"):
    with h5py.File(cache, "r") as f:
        present = np.asarray(f["row"])
    missing = np.setdiff1d(rows, present)
    extra = np.setdiff1d(present, rows)
    print(f"[188] {cache}: {len(present)} rows, {len(missing)} of the cohort missing, "
          f"{len(extra)} not in the cohort")
    if len(extra):
        raise SystemExit(f"[188] {cache} holds rows outside the committed cohort")
PYEOF

COMMON=(
  --latents   "$LEGACY_LATENTS,$BW_LATENTS"
  --blockouts "$LEGACY_BLOCKOUTS,$BW_BLOCKOUTS"
  --region_free
  --batch 8 --lr 1e-4 --width 512 --depth 8 --heads 8
  --pair_frac 0.8 --pair_t_min 0.35 --cfg_drop 0.1
  --seed 0 --log_every 1000 --save_every 5000
)

# `train_vecset.py --resume` refuses a checkpoint already at or past --steps, which is correct for
# it and wrong for a re-run of this script. Read the step and skip a finished phase instead.
step_of() {
  [ -f "$1" ] || { echo 0; return; }
  "${PY[@]}" -c "import torch,sys; print(int(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['step']))" "$1" 2>/dev/null || echo 0
}

echo "[188] git $(git rev-parse --short HEAD)  dirty=$(git status --porcelain | wc -l)"

PHASE1_STEP=$(step_of "$PHASE1/vecset_denoiser.pth")
if [ "$PHASE1_STEP" -ge 180000 ]; then
  echo "[188] phase 1 already at step $PHASE1_STEP -- skipping"
else
  echo "[188] phase 1 (surface term OFF, $PHASE1_STEP -> 180000) starting $(date -u '+%F %T UTC')"
  RESUME1=()
  [ "$PHASE1_STEP" -gt 0 ] && RESUME1=(--resume "$PHASE1/vecset_denoiser.pth")
  "${PY[@]}" "$TRAIN" "${COMMON[@]}" \
    --steps 180000 --surf_weight 0.0 --archive_every 20000 \
    "${RESUME1[@]}" --out "$PHASE1"
fi

PHASE2_STEP=$(step_of "$PHASE2/vecset_denoiser.pth")
if [ "$PHASE2_STEP" -ge 240000 ]; then
  echo "[188] phase 2 already at step $PHASE2_STEP -- skipping"
else
  # Resume from phase 2's own checkpoint if it has one, otherwise hand over from phase 1. The
  # handover is what makes this the frozen source's lineage rather than a fresh 60k-step run.
  if [ "$PHASE2_STEP" -gt 0 ]; then RESUME2="$PHASE2/vecset_denoiser.pth"
  else RESUME2="$PHASE1/vecset_denoiser.pth"; fi
  echo "[188] phase 2 (surface term ON, $PHASE2_STEP -> 240000) from $RESUME2 at $(date -u '+%F %T UTC')"
  "${PY[@]}" "$TRAIN" "${COMMON[@]}" \
    --steps 240000 --surf_weight 1.0 --archive_every 10000 \
    --resume "$RESUME2" --out "$PHASE2"
fi

echo "[188] RETRAIN COMPLETE $(date -u '+%F %T UTC') -> $PHASE2/vecset_denoiser.pth"
echo "[188] next: score against the pre-registered bar, then freeze and hash. See"
echo "[188] docs/wayfinding/whole-volume-voxel-transform/188-a2-retrain.md"
