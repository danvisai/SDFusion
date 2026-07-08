#!/usr/bin/env bash
# Launch the GenerativeTowns web demo (FastAPI + three.js) on the GPU box.
#   bash scripts/server/run_web_demo.sh [PORT]
# Then from your laptop:  ssh -L <PORT>:<this-host>:<PORT> dsimhadr@gilbreth.rcac.purdue.edu
# and open  http://localhost:<PORT>/
set -e
cd "$(dirname "$0")/../.."
PORT="${1:-8099}"
echo "[web demo] host=$(hostname)  port=$PORT  ->  http://localhost:$PORT/ (after tunneling)"
# PYTHONPYCACHEPREFIX: keep .pyc I/O on node-local disk — Lustre scratch intermittently
# stalls single-file reads (memory/env_gilbreth_lustre_file_stall.md) and a stalled .pyc
# read hangs a lazy import (and the request behind it) forever at 0% CPU.
mkdir -p /tmp/pyc_dsimhadr
exec env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
  HDF5_USE_FILE_LOCKING=FALSE PYTHONPYCACHEPREFIX=/tmp/pyc_dsimhadr \
  ./sdfusion/bin/python -m uvicorn scripts.server.inference_service:app \
    --host 0.0.0.0 --port "$PORT" --log-level warning
