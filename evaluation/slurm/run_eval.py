#!/usr/bin/env python3
"""Supervise one broker/server and a bounded queue of simulation clients."""
import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
ROBOTWIN_TASKS = """adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size
click_alarmclock click_bell dump_bin_bigbin grab_roller handover_block handover_mic
hanging_mug lift_pot move_can_pot move_pillbottle_pad move_playingcard_away move_stapler_pad
open_laptop open_microwave pick_diverse_bottles pick_dual_bottles place_a2b_left place_a2b_right
place_bread_basket place_bread_skillet place_burger_fries place_can_basket place_cans_plasticbox
place_container_plate place_dual_shoes place_empty_cup place_fan place_mouse_pad
place_object_basket place_object_scale place_object_stand place_phone_stand place_shoe
press_stapler put_bottles_dustbin put_object_cabinet rotate_qrcode scan_object shake_bottle
shake_bottle_horiz stack_blocks_three stack_blocks_two stack_bowls_three stack_bowls_two
stamp_seal turn_switch""".split()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("benchmark", choices=("robocasa", "robotwin"))
    p.add_argument("--tasks", nargs="+", help="RoboCasa indices 0–23 or RoboTwin task names; default: all")
    p.add_argument("--num-evals", type=int, default=100)
    p.add_argument("--clients", type=int, default=2)
    p.add_argument("--exp-path", type=Path)
    p.add_argument("--wan-checkpoint-dir", type=Path, default=ROOT / "checkpoints/wan22_5b")
    p.add_argument("--steps", default="last")
    p.add_argument("--denoise-steps", type=int, default=50)
    p.add_argument("--action-denoise-steps", type=int, default=10)
    p.add_argument("--action-length", type=int, default=32)
    p.add_argument("--cfg", type=float, default=0.0)
    p.add_argument("--task-config", default="demo_randomized")
    p.add_argument("--seed", type=int, default=0, help="RoboTwin seed (RoboCasa uses upstream per-rank seeds)")
    p.add_argument("--save-root-dir", type=Path)
    p.add_argument("--frontend-port", type=int)
    p.add_argument("--backend-port", type=int)
    p.add_argument("--startup-timeout", type=float, default=1800)
    p.add_argument("--task-timeout", type=float, default=0, help="Seconds per client, including inference; 0 disables")
    p.add_argument("--dry-run", action="store_true", help="Print commands without loading packages or starting processes")
    args = p.parse_args()
    if min(args.num_evals, args.clients, args.action_length, args.denoise_steps) < 1:
        p.error("num-evals, clients, action-length, and denoise-steps must be positive")
    if args.startup_timeout <= 0 or args.task_timeout < 0 or args.action_denoise_steps < 0:
        p.error("Invalid timeout or action-denoise-steps")
    if args.benchmark == "robotwin" and args.action_length < 4:
        p.error("RoboTwin video recording requires action-length >= 4")
    for port in (args.frontend_port, args.backend_port):
        if port is not None and not 1024 <= port <= 65535:
            p.error("Ports must be in 1024–65535")
    if args.frontend_port is not None and args.frontend_port == args.backend_port:
        p.error("Frontend and backend ports must differ")
    tasks = args.tasks or (list(map(str, range(24))) if args.benchmark == "robocasa" else ROBOTWIN_TASKS)
    valid = set(map(str, range(24))) if args.benchmark == "robocasa" else set(ROBOTWIN_TASKS)
    if len(tasks) != len(set(tasks)) or any(task not in valid for task in tasks):
        p.error("Tasks must be unique valid indices/names for the selected benchmark")
    args.tasks = tasks
    args.exp_path = (args.exp_path or ROOT / "checkpoints" / f"{args.benchmark}_sft").resolve()
    args.wan_checkpoint_dir = args.wan_checkpoint_dir.resolve()
    run_id = os.environ.get("SLURM_JOB_ID", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    args.save_root_dir = (args.save_root_dir or ROOT / "eval_results" / args.benchmark / run_id).resolve()
    return args


def commands(args, frontend, backend):
    python = sys.executable
    broker = [python, "-u", str(ROOT / "evaluation/policy_broker.py"),
              "--host", "127.0.0.1", "--frontend-port", str(frontend), "--backend-port", str(backend)]
    server = [python, "-u", str(ROOT / "evaluation/policy_server.py"),
              "--exp-path", str(args.exp_path), "--steps", args.steps,
              "--wan-checkpoint-dir", str(args.wan_checkpoint_dir),
              "--broker-addr", "127.0.0.1", "--broker-port", str(backend),
              "--denoise-steps", str(args.denoise_steps), "--action-denoise-steps", str(args.action_denoise_steps)]
    clients = {}
    for task in args.tasks:
        cmd = [python, "-u", str(ROOT / "evaluation" / f"{args.benchmark}_client.py"),
               "--server-addr", "127.0.0.1", "--server-port", str(frontend),
               "--num-evals-per-worker", str(args.num_evals),
               "--save-root-dir", str(args.save_root_dir),
               "--action-length", str(args.action_length), "--cfg", str(args.cfg)]
        if args.benchmark == "robocasa":
            # The upstream client waits for world_size files before exiting.
            # With a bounded queue, a global barrier would prevent queued tasks from starting.
            cmd += ["--env-global-rank", task, "--world-size", "1"]
        else:
            cmd += ["--task-name", task, "--task-config", args.task_config, "--seed", str(args.seed)]
        clients[task] = cmd
    return {"broker": broker, "server": server, "clients": clients}


def reserve_port(port):
    sock = socket.socket()
    sock.bind(("127.0.0.1", port or 0))
    return sock


def main():
    args = parse_args()
    if args.dry_run:
        print(json.dumps(commands(args, args.frontend_port or 10086, args.backend_port or 10087), indent=2))
        return
    os.chdir(ROOT)
    os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "evaluation"), os.environ.get("PYTHONPATH", "")])
    subprocess.run([sys.executable, str(ROOT / "evaluation/slurm/check_env.py"),
                    "--benchmark", args.benchmark, "--require-gpu", "--exp-path", str(args.exp_path),
                    "--wan-checkpoint-dir", str(args.wan_checkpoint_dir), "--steps", args.steps], check=True)
    if args.save_root_dir.exists() and any(args.save_root_dir.iterdir()):
        raise RuntimeError(f"Refusing to mix results with an existing run: {args.save_root_dir}")
    logs = args.save_root_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with reserve_port(args.frontend_port) as front, reserve_port(args.backend_port) as back:
        cmds = commands(args, front.getsockname()[1], back.getsockname()[1])
    manifest = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "commands": cmds, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "host": socket.gethostname()}
    (args.save_root_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    children = []
    handles = []

    def spawn(name, cmd):
        handle = (logs / f"{name}.log").open("w")
        handles.append(handle)
        proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(proc)
        print(f"Started {name} (pid {proc.pid}); log: {handle.name}", flush=True)
        return proc

    def interrupted(signum, frame):
        raise InterruptedError(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        broker = spawn("broker", cmds["broker"])
        server = spawn("server", cmds["server"])

        def check_services():
            for name, proc in (("broker", broker), ("server", server)):
                if proc.poll() is not None:
                    raise RuntimeError(f"{name} exited with {proc.returncode}; see {logs / (name + '.log')}")

        deadline = time.monotonic() + args.startup_timeout
        while True:
            check_services()
            # READY is emitted by the broker only after the server loads the model.
            if " READY " in (logs / "broker.log").read_text():
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"Policy startup timed out; see {logs / 'server.log'}")
            time.sleep(1)
        pending = deque(args.tasks)
        active = {}
        while pending or active:
            check_services()
            while pending and len(active) < args.clients:
                task = pending.popleft()
                active[task] = (spawn(f"client-{task}", cmds["clients"][task]), time.monotonic())
            for task, (proc, started) in list(active.items()):
                status = proc.poll()
                if status is not None:
                    if status:
                        raise RuntimeError(f"Task {task} exited with {status}; see {logs / ('client-' + task + '.log')}")
                    print(f"Completed task {task}", flush=True)
                    del active[task]
                elif args.task_timeout and time.monotonic() - started > args.task_timeout:
                    raise TimeoutError(f"Task {task} timed out; see {logs / ('client-' + task + '.log')}")
            time.sleep(1)
        subprocess.run([sys.executable, str(ROOT / "evaluation/merge_results.py"), str(args.save_root_dir)], check=True)
        if not (args.save_root_dir / "eval_summary.json").is_file():
            raise RuntimeError("Clients exited without producing an evaluation summary")
        print(f"Evaluation complete: {args.save_root_dir / 'eval_summary.json'}", flush=True)
    finally:
        # Also terminate descendants (e.g. RoboTwin motion planners).
        for proc in children:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 10
        for proc in children:
            try:
                proc.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        for proc in children:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, TimeoutError, InterruptedError, subprocess.CalledProcessError, OSError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        sys.exit(1)
