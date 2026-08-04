#!/usr/bin/env bash
# Gilbreth-friendly wrapper for the BuildingNet SDF regeneration.
# - strips XALT preload (breaks the bundled binary)
# - sets LD_LIBRARY_PATH to find libtcmalloc, libtbb, and bundled libs
# - chmod's the binaries in case they were re-checked-out
# - uses our sdfusion venv python
#
# Run from preprocess/ (the launcher cd's there):
#     ./launchers/launch_create_sdf_building_gilbreth.sh
set -e

cd "$(dirname "$0")/.."   # cd into preprocess/
chmod +x isosurface/computeDistanceField isosurface/computeMarchingCubes 2>/dev/null || true

# Build LD_LIBRARY_PATH from the LIB_PATH file (which was the original mechanism)
ISO="$(pwd)/isosurface"
LIB_LD="$ISO:$ISO/tbb/tbb2018_20180822oss/lib/intel64/gcc4.7"

PYTHON="${PYTHON:-$(pwd)/../sdfusion/bin/python}"

dset='building'
reduce=4
thread_num=${THREAD_NUM:-9}

echo "[gilbreth-cs] python:    ${PYTHON}"
echo "[gilbreth-cs] dset:      ${dset}"
echo "[gilbreth-cs] reduce:    ${reduce}  (output 64^3)"
echo "[gilbreth-cs] threads:   ${thread_num}"
echo "[gilbreth-cs] LD_LIBRARY_PATH: ${LIB_LD}"
echo

env -u LD_PRELOAD -u LD_LIBRARY_PATH \
    LD_LIBRARY_PATH="$LIB_LD" \
    "$PYTHON" -u create_sdf.py \
      --dset ${dset} \
      --thread_num ${thread_num} \
      --reduce ${reduce}
