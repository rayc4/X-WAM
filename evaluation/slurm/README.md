# Slurm evaluation

Run commands from the X-WAM repository root. The top-level `slurm` symlink is a
read-only reference for the `gb10` partition, GPU request, and Slurm invocation
style. All scripts and caches for this workflow live in X-WAM.

## Environment and assets

```bash
bash evaluation/slurm/setup.sh robocasa
bash evaluation/slurm/download.sh robocasa
```

`setup.sh` initializes the pinned submodules and creates `.venv` with Python 3.10
installed under `.cache/python` on shared project storage.
It installs evaluation dependencies, simulator packages, and RoboCasa kitchen
assets. It excludes DeepSpeed, training datasets, and training data readers.
`download.sh` fetches the selected fine-tuned checkpoint and Wan2.2-TI2V-5B base
weights, including the text encoder and tokenizer. RoboCasa weights and base
weights require approximately 73 GB of disk space. Downloads are resumable by
rerunning the command. `download.sh all` fetches both fine-tuned checkpoints.

ARM/GB10 uses PyTorch 2.9.0, TorchVision 0.24.0, and CUDA 13 wheels. Other Linux
machines default to the upstream example, PyTorch 2.8.0 / CUDA 12.9. The
evaluation stack pins NumPy 1.23.5 and Transformers 4.51.3. RoboCasa's legacy
NumPy/Numba pins conflict with ARM wheels. RoboCasa also asserts MuJoCo 3.2.6 at
import time, while the pinned RoboSuite metadata requests >=3.3.0. The setup
installs both pinned simulator packages with `--no-deps` and supplies NumPy
1.23.5, Numba 0.60.0, and MuJoCo 3.2.6 explicitly. `uv pip check` will report these
known simulator metadata conflicts; use `check_env.py` and a smoke evaluation
to check runtime compatibility.

FlashAttention is optional: the repository has a PyTorch SDPA fallback and uses
FlexAttention for masked attention. To build FlashAttention explicitly:

```bash
XWAM_INSTALL_FLASH_ATTN=1 bash evaluation/slurm/setup.sh robocasa
```

Use `XWAM_PYTHON=/path/to/python3.10` to choose the interpreter when creating the
environment. `PYTHON_BIN=/path/to/env/bin/python` selects an existing environment.
`XWAM_TORCH_VERSION`, `XWAM_TORCHVISION_VERSION`, and `XWAM_TORCH_INDEX` override
the Torch installation. `XWAM_ROBOCASA_ASSETS=/path/to/v0.2/models/assets` lets
setup copy compatible local simulator assets instead of downloading them.
Nothing is installed into other projects or the referenced Slurm directory.

## Submit jobs

```bash
# First run: task 0, one episode
bash evaluation/slurm/submit.sh --time=01:00:00 -- robocasa --tasks 0 --num-evals 1 --clients 1

# All 24 tasks, 100 episodes each (defaults)
bash evaluation/slurm/submit.sh robocasa

# Subset, fewer episodes, different Slurm wall time
bash evaluation/slurm/submit.sh --time=04:00:00 -- robocasa --tasks 0 4 23 --num-evals 10 --clients 2

# Inspect generated commands without starting processes or allocating a GPU
.venv/bin/python evaluation/slurm/run_eval.py robocasa --tasks 0 --num-evals 1 --dry-run
```

SBATCH options go before `--`; evaluation options go after the benchmark.
The default job requests one node, one GPU, 8 CPUs, 96 GB RAM, and 24 hours on
`gb10`. The policy contains a 5B video backbone and large text encoder; the
reference script's 16 GB request is insufficient for loading these weights.
Only one policy server runs per allocation. Keep `--gres=gpu:1` and `--nodes=1`.
Slurm's `CUDA_VISIBLE_DEVICES` is preserved for inference and rendering.

Two simulation clients run at a time by default; tasks beyond that wait in a
queue. This limits CPU and memory use while sharing the GPU with inference.
`--clients 1` is useful for troubleshooting. The launcher uses `--world-size 1`
for each RoboCasa client to avoid its global completion barrier deadlocking
the task queue. Task indices and upstream rollout seeds are preserved.

The broker binds only to localhost on automatically chosen ports. Clients start
after the broker receives the server's READY message. A failed service or client
causes the job to fail and terminates the other processes. Model loading has a
30-minute timeout; `--startup-timeout` changes it. `--task-timeout SECONDS` bounds
each client's lifetime including time spent waiting for inference. Slurm wall
time bounds the whole job. Checkpoints and environment imports are checked
before starting services.

Use `--exp-path`, `--wan-checkpoint-dir`, and `--steps` for alternate local
checkpoints. Denoising defaults match the evaluation README: 50 video steps and
10 action steps. Use `--denoise-steps` and `--action-denoise-steps` to override.
`--action-length` defaults to 32; `--cfg` defaults to 0.

## Logs and results

Scheduler output is in `eval_results/slurm_logs/xwam-eval-<job_id>.out` and `.err`.
Each run writes:

```text
eval_results/robocasa/<job_id>/
  run.json                 # exact commands, arguments, node, and GPU visibility
  logs/broker.log
  logs/server.log
  logs/client-<task>.log
  <task_name>/*.mp4
  eval_results_<rank>.json
  eval_summary.json
```

`--save-root-dir /absolute/path` overrides the run directory. Existing nonempty
directories are rejected to prevent mixing results. `eval_summary.json` is
generated after all clients succeed. A failed run can still contain partial
results; inspect its client logs before interpreting any aggregate.

```bash
squeue -u "$USER"
tail -f eval_results/slurm_logs/xwam-eval-<job_id>.out
scancel <job_id>

# Import checks on the login node (CUDA execution needs an allocation)
source evaluation/slurm/env.sh
"$PYTHON_BIN" evaluation/slurm/check_env.py --benchmark robocasa

# Inside an allocated GPU shell, also check CUDA SDPA/FlexAttention
"$PYTHON_BIN" evaluation/slurm/check_env.py --benchmark robocasa --require-gpu
```

## RoboTwin

The pinned SAPIEN 3.0.0b1 and MPLib 0.2.1 distributions offer Linux x86_64 wheels,
with no ARM wheels. The `gb10` partition is ARM, so its RoboTwin client cannot be
installed with these packages. `setup.sh robotwin` and the job's preflight check
report this explicitly. Building/porting native simulator packages is outside
this setup.

On an x86_64 cluster, first create the base environment using `setup.sh base`,
then complete the native simulator and asset setup in
`third_party/RoboTwin/README.md`. Retain the inference Torch/Hugging Face versions
and NumPy 1.23.5; use a separate client environment if simulator dependencies
conflict. The current single-allocation launcher expects policy and simulator
dependencies in the same environment. Validate it with
`check_env.py --benchmark robotwin` before submitting:

```bash
bash evaluation/slurm/download.sh robotwin
bash evaluation/slurm/submit.sh --partition=<x86_gpu_partition> -- robotwin --tasks adjust_bottle --num-evals 1 --clients 1

# All 50 tasks, randomized setting
bash evaluation/slurm/submit.sh --partition=<x86_gpu_partition> -- robotwin

# Clean setting
bash evaluation/slurm/submit.sh --partition=<x86_gpu_partition> -- robotwin --task-config demo_clean
```

The launcher runs from the repository root and exposes `evaluation/X-WAM` via
`PYTHONPATH`, so there is no need to copy policy files into RoboTwin or edit its
root in `robotwin_client.py`. `--seed` and `--task-config` select RoboTwin seed and
setting. RoboTwin result paths are absolute to account for the client's `chdir`.
