#!/usr/bin/env python3
"""Check evaluation imports, optional GPU execution, and local model assets."""
import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "evaluation")]


def check_assets(exp_path, wan_path, steps):
    from omegaconf import OmegaConf
    config_file = exp_path / "config.yaml"
    checkpoint = exp_path / "checkpoints" / f"{steps}.ckpt/checkpoint/mp_rank_00_model_states.pt"
    required = [config_file, checkpoint, wan_path / "config.json"]
    if config_file.is_file():
        config = OmegaConf.load(config_file)
        required.extend(wan_path / config[key] for key in ("t5_checkpoint", "vae_checkpoint"))
        tokenizer = wan_path / config.t5_tokenizer
        if not tokenizer.is_dir():
            raise RuntimeError(f"Missing tokenizer directory: {tokenizer}")
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    indices = list(wan_path.glob("*.safetensors.index.json"))
    if indices:
        for index in indices:
            weights = set(json.loads(index.read_text())["weight_map"].values())
            missing.extend(str(wan_path / name) for name in weights if not (wan_path / name).is_file())
    elif not list(wan_path.glob("diffusion_pytorch_model*.safetensors")):
        missing.append(str(wan_path / "diffusion_pytorch_model*.safetensors"))
    if missing:
        raise RuntimeError("Missing model assets:\n" + "\n".join(missing) +
                           "\nRun bash evaluation/slurm/download.sh <benchmark>.")
    print("Model assets: OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("robocasa", "robotwin"))
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--exp-path", type=Path)
    parser.add_argument("--wan-checkpoint-dir", type=Path, default=ROOT / "checkpoints/wan22_5b")
    parser.add_argument("--steps", default="last")
    args = parser.parse_args()
    print(f"Python {platform.python_version()}; {platform.machine()}; interpreter {sys.executable}")
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("The simulation environments require Python 3.10")
    for name in ("torch", "torchvision", "numpy", "scipy", "zmq", "tyro", "imageio_ffmpeg",
                 "diffusers", "transformers", "lightning", "omegaconf"):
        module = importlib.import_module(name)
        print(f"{name}: {getattr(module, '__version__', 'OK')}")
    import numpy as np
    if np.__version__ != "1.23.5":
        raise RuntimeError(f"Expected numpy==1.23.5, got {np.__version__}")
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-version"], check=True, stdout=subprocess.DEVNULL)
    if args.benchmark == "robotwin" and platform.machine() != "x86_64":
        raise RuntimeError("RoboTwin's pinned SAPIEN/MPLib wheels require Linux x86_64. "
                           "Use an x86_64 Slurm node/environment for this benchmark.")
    if args.benchmark:
        # Upstream RoboTwin resolves relative simulator paths from the repo root.
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "evaluation"), env.get("PYTHONPATH", "")])
        subprocess.run([sys.executable, str(ROOT / "evaluation" / f"{args.benchmark}_client.py"), "--help"],
                       cwd=ROOT, env=env, stdout=subprocess.DEVNULL, check=True)
        print(f"{args.benchmark} client imports: OK")
    if args.exp_path:
        check_assets(args.exp_path, args.wan_checkpoint_dir, args.steps)
    if args.require_gpu:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; run this check inside a GPU Slurm allocation")
        # T5EncoderModel's default argument initializes CUDA at import time.
        # Check policy imports only inside an actual GPU allocation.
        importlib.import_module("runners.xwam_runner")
        from modules.attention import attention
        print(f"GPU: {torch.cuda.get_device_name(0)}; CUDA {torch.version.cuda}")
        q = torch.randn(1, 32, 2, 64, dtype=torch.bfloat16, device="cuda")
        for mask in (None, lambda b, h, q_idx, kv_idx: q_idx >= kv_idx):
            result = attention(q, q, q, attn_mask_func=mask)
            assert result.shape == q.shape and torch.isfinite(result).all()
        torch.cuda.synchronize()
        print("CUDA attention and FlexAttention: OK")
    else:
        print("Policy runner import/CUDA checks require --require-gpu inside a GPU allocation")
    print("EVAL_ENV_OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Environment check failed: {error}", file=sys.stderr)
        sys.exit(1)
