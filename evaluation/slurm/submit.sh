#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
usage() {
    echo "Usage: bash evaluation/slurm/submit.sh [SBATCH_OPTIONS --] {robocasa|robotwin} [EVAL_OPTIONS]"
    echo "Example: bash evaluation/slurm/submit.sh --time=01:00:00 -- robocasa --tasks 0 --num-evals 1 --clients 1"
    echo "Run .venv/bin/python evaluation/slurm/run_eval.py --help for evaluation options."
}
if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
sbatch_args=()
if [[ ${1:-} == -* ]]; then
    while (( $# > 0 )) && [[ $1 != -- ]]; do sbatch_args+=("$1"); shift; done
    if (( $# == 0 )); then usage >&2; exit 2; fi
    shift
fi
if [[ ${1:-} != robocasa && ${1:-} != robotwin ]]; then usage >&2; exit 2; fi
source "$root/evaluation/slurm/env.sh"
if [[ ! -x $PYTHON_BIN ]]; then
    echo "Missing environment; run bash evaluation/slurm/setup.sh robocasa (or robotwin)." >&2
    exit 1
fi
# Validate all evaluation arguments before spending an allocation.
"$PYTHON_BIN" "$root/evaluation/slurm/run_eval.py" "$@" --dry-run >/dev/null
mkdir -p "$root/eval_results/slurm_logs"
exec sbatch --chdir="$root" \
    --output="$root/eval_results/slurm_logs/%x-%j.out" \
    --error="$root/eval_results/slurm_logs/%x-%j.err" \
    "${sbatch_args[@]}" "$root/evaluation/slurm/evaluate.sbatch" "$@"
