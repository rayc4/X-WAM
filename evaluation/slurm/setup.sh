#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
benchmark=${1:-robocasa}
if [[ $benchmark != base && $benchmark != robocasa && $benchmark != robotwin ]]; then
    echo "Usage: bash evaluation/slurm/setup.sh [base|robocasa|robotwin]" >&2; exit 2
fi
if [[ $benchmark == robotwin && $(uname -m) != x86_64 ]]; then
    echo "RoboTwin requires Linux x86_64 SAPIEN/MPLib wheels; gb10 ARM nodes are unsupported." >&2
    echo "Use setup.sh robocasa here; follow the pinned RoboTwin README on an x86_64 cluster." >&2
    exit 1
fi
cd "$XWAM_ROOT"
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$HF_HOME"
command -v uv >/dev/null || { echo "Install uv before running setup." >&2; exit 1; }
git submodule update --init --recursive
if [[ ! -x $PYTHON_BIN ]]; then
    if [[ -n ${XWAM_PYTHON:-} ]]; then
        eval_python=$XWAM_PYTHON
    else
        # Put the interpreter on shared storage rather than a login-node home.
        uv python install 3.10 --no-bin --install-dir "$UV_PYTHON_INSTALL_DIR"
        eval_python=$(uv python find --managed-python 3.10)
    fi
    uv venv --python "$eval_python" "$(dirname -- "$(dirname -- "$PYTHON_BIN")")"
fi
if [[ $(uname -m) == aarch64 ]]; then
    torch_version=${XWAM_TORCH_VERSION:-2.9.0}
    torchvision_version=${XWAM_TORCHVISION_VERSION:-0.24.0}
    torch_index=${XWAM_TORCH_INDEX:-https://download.pytorch.org/whl/cu130}
else
    torch_version=${XWAM_TORCH_VERSION:-2.8.0}
    torchvision_version=${XWAM_TORCHVISION_VERSION:-0.23.0}
    torch_index=${XWAM_TORCH_INDEX:-https://download.pytorch.org/whl/cu129}
fi
uv pip install --python "$PYTHON_BIN" "torch==$torch_version" "torchvision==$torchvision_version" --index-url "$torch_index"
uv pip install --python "$PYTHON_BIN" -r evaluation/requirements.txt
if [[ $benchmark == robocasa ]]; then
    uv pip install --python "$PYTHON_BIN" -r evaluation/requirements-robocasa.txt
    # Install pinned submodule code without applying its conflicting legacy pins.
    uv pip install --python "$PYTHON_BIN" --no-deps -e third_party/robosuite -e third_party/robocasa
    "$PYTHON_BIN" evaluation/slurm/setup_robocasa_assets.py
elif [[ $benchmark == robotwin ]]; then
    echo "Base environment installed. Complete simulator/native asset setup from third_party/RoboTwin/README.md."
    echo "Keep the base Torch/Hugging Face versions and numpy==1.23.5. Then run check_env.py --benchmark robotwin."
    exit 1
fi
# FlashAttention is optional: modules/attention.py has a native PyTorch fallback.
# Building it is explicit because ARM/CUDA builds are expensive.
if [[ ${XWAM_INSTALL_FLASH_ATTN:-0} == 1 ]]; then
    uv pip install --python "$PYTHON_BIN" ninja wheel psutil
    uv pip install --python "$PYTHON_BIN" flash-attn==2.8.3 --no-build-isolation
fi
check_args=()
if [[ $benchmark != base ]]; then check_args+=(--benchmark "$benchmark"); fi
"$PYTHON_BIN" evaluation/slurm/check_env.py "${check_args[@]}"
uv pip freeze --python "$PYTHON_BIN" > .cache/evaluation-installed.txt
echo "Environment ready: $PYTHON_BIN"
echo "Download evaluation weights: bash evaluation/slurm/download.sh ${benchmark/base/robocasa}"
