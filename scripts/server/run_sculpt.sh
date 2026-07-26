#!/usr/bin/env bash
# Launch the inference service that serves the unbound.io-style SDF sculptor.
#
#   web/sculpt.html  -> /building_sdf (64^3 SDF volume) + /snap_sdf (generative SDEdit snap)
#
# NOTE: the Bash sandbox kills network-binding servers (exit 144) -> run this OUTSIDE the
# sandbox (or via `! scripts/server/run_sculpt.sh` in the Claude prompt). Then tunnel:
#
#   ssh -L 8099:<gpu-node>:8099 <you>@gilbreth.rcac.purdue.edu
#   open http://localhost:8099/sculpt.html
#
set -euo pipefail
cd "$(dirname "$0")/../.."          # repo root
PORT="${1:-8099}"
exec env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. HDF5_USE_FILE_LOCKING=FALSE \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  ./sdfusion/bin/python -m uvicorn scripts.server.inference_service:app \
  --host 0.0.0.0 --port "$PORT"
