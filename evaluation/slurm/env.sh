#!/usr/bin/env bash
# Source from setup scripts and Slurm jobs; keep all writes in this project.
XWAM_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export XWAM_ROOT
export PYTHON_BIN="${PYTHON_BIN:-$XWAM_ROOT/.venv/bin/python}"
export PATH="$(dirname -- "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$XWAM_ROOT:$XWAM_ROOT/evaluation${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$XWAM_ROOT/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$XWAM_ROOT/.cache/python}"
export HF_HOME="${HF_HOME:-$XWAM_ROOT/.cache/huggingface}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${XWAM_THREADS:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMBA_NUM_THREADS="$OMP_NUM_THREADS"
# CUDA wheels provide their own native libraries.
for xwam_lib in "$(dirname -- "$PYTHON_BIN")"/../lib/python*/site-packages/nvidia/*/lib; do
    if [[ -d $xwam_lib ]]; then
        export LD_LIBRARY_PATH="$xwam_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done
unset xwam_lib
