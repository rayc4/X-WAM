#!/usr/bin/env bash
#SBATCH --job-name=xwam-eval
#SBATCH --partition=gb10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@60

set -euo pipefail
# submit.sh supplies an absolute working directory (Slurm spools this file).
source "$PWD/evaluation/slurm/env.sh"
export TMPDIR="$XWAM_ROOT/.cache/tmp/${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor"
export TRITON_CACHE_DIR="$TMPDIR/triton"
mkdir -p "$TMPDIR" "$HF_HOME"
date -Is
hostname
printf 'JOB=%s NODES=%s GPU=%s\n' "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST" "${CUDA_VISIBLE_DEVICES:-unset}"
git rev-parse HEAD
# exec lets Slurm's batch-shell TERM signal reach the process supervisor.
exec srun --kill-on-bad-exit=1 "$PYTHON_BIN" -u evaluation/slurm/run_eval.py "$@"
