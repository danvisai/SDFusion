#!/usr/bin/env bash
# Wait for the GPU, then run #92 arms through the tracked definition in run_aligned_retrain.py.
#
# One box here holds one arm: the surface loss reserves ~37 GB of the 64 GB card, so a second
# trainer cannot start while one is running. On a multi-GPU host this wrapper is unnecessary --
# call run_aligned_retrain.py directly, one process per device.
#
#   launchers/queue_issue92_arm.sh N        # the NL+DE-only probe
#   launchers/queue_issue92_arm.sh B        # the candidate
#
# Run it detached so a logout cannot take it down; a SIGHUP already killed one run at 08:50 on
# 2026-08-16 and cost ~10k steps:
#   tmux new-session -d -s issue92q "launchers/queue_issue92_arm.sh N"

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYBIN="$REPO/venv/bin/python"
TRAIN="$REPO/scripts/train_vecset.py"
RUNNER="$REPO/scripts/foundations/run_aligned_retrain.py"

[ $# -ge 1 ] || { echo "usage: $(basename "$0") ARM [ARM...]" >&2; exit 2; }

# Match the trainer by exact "python <script>" cmdline prefix, so neither this wrapper nor the
# DataLoader forkserver children can be mistaken for a running arm.
trainer_running() {
  local p c
  for p in $(pgrep -f train_vecset 2>/dev/null); do
    c=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
    case "$c" in "$PYBIN $TRAIN "*) return 0 ;; esac
  done
  return 1
}

cd "$REPO"
if trainer_running; then
  echo "[queue] a trainer holds the GPU; waiting since $(date -u '+%F %T UTC')"
  while trainer_running; do sleep 120; done
fi
echo "[queue] GPU free at $(date -u '+%F %T UTC'); git $(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
exec "$PYBIN" "$RUNNER" --arms "$@"
