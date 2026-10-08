#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
benchmark=${1:-robocasa}
if [[ $benchmark != robocasa && $benchmark != robotwin && $benchmark != all ]]; then
    echo "Usage: bash evaluation/slurm/download.sh [robocasa|robotwin|all]" >&2; exit 2
fi
cd "$XWAM_ROOT"
mkdir -p "$HF_HOME"
# Only fine-tuned evaluation weights; omit pretraining weights and datasets.
"$PYTHON_BIN" - "$benchmark" <<'PY'
import sys
from huggingface_hub import snapshot_download
benchmarks = ('robocasa', 'robotwin') if sys.argv[1] == 'all' else (sys.argv[1],)
for benchmark in benchmarks:
    snapshot_download('sharinka0715/X-WAM-checkpoints', local_dir='checkpoints',
                      allow_patterns=[f'{benchmark}_sft/**'], max_workers=4)
snapshot_download('Wan-AI/Wan2.2-TI2V-5B', local_dir='checkpoints/wan22_5b', max_workers=4)
PY
