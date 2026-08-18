"""Run finetune only when GPU has been idle long enough; resume after manual stop.

Example (background):
  nohup python -m finetune_ashare.idle_train_runner \\
    --config finetune_ashare/configs/mainboard_daily.yaml \\
    --mem-threshold 0.5 --idle-minutes 30 \\
    > finetune_ashare/outputs/idle_runner.log 2>&1 &
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

from finetune_ashare.config_loader import AshareFinetuneConfig


def _log(msg: str, *, log_file: str | None) -> None:
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _query_gpu(gpu_id: int) -> tuple[int, int, int]:
    """Return (mem_used_mib, mem_total_mib, gpu_util_percent)."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu_id),
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(f"nvidia-smi failed: {exc}") from exc

    parts = [p.strip() for p in out.split(",")]
    if len(parts) != 3:
        raise RuntimeError(f"unexpected nvidia-smi output: {out!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _is_idle(
    gpu_id: int,
    *,
    mem_threshold: float,
    util_threshold: float,
) -> tuple[bool, str]:
    mem_used, mem_total, util = _query_gpu(gpu_id)
    mem_ratio = mem_used / mem_total if mem_total > 0 else 1.0
    ok = mem_ratio < mem_threshold and util <= util_threshold
    detail = (
        f"gpu={gpu_id} mem={mem_used}/{mem_total}MiB ({mem_ratio:.1%}) "
        f"util={util}% "
        f"(need mem<{mem_threshold:.0%} and util<={util_threshold:.0f}%)"
    )
    return ok, detail


def _build_train_cmd(config_path: str) -> tuple[list[str], str]:
    config = AshareFinetuneConfig(config_path)
    cmd = [sys.executable, "-m", "finetune_ashare", "--config", config_path]
    if os.path.isfile(config.basemodel_last_train_path):
        cmd.append("--resume-predictor")
        mode = "resume-predictor"
    elif os.path.isdir(config.tokenizer_best_path) and os.path.isfile(
        os.path.join(config.tokenizer_best_path, "config.json")
    ):
        cmd.append("--skip-tokenizer")
        mode = "skip-tokenizer"
    else:
        mode = "full"
    return cmd, mode


def run(args: argparse.Namespace) -> int:
    log_file = args.log_file
    _log(
        f"Idle runner started: mem<{args.mem_threshold:.0%}, util<={args.util_threshold}%, "
        f"idle={args.idle_minutes}min, poll={args.poll_seconds}s, gpu={args.gpu_id}",
        log_file=log_file,
    )

    child: subprocess.Popen | None = None

    def _terminate_child(*_exc: object) -> None:
        nonlocal child
        if child is not None and child.poll() is None:
            _log(f"Stopping training pid={child.pid} ...", log_file=log_file)
            child.send_signal(signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            _log("Training stopped.", log_file=log_file)
        child = None

    def _handle_signal(signum: int, _frame: object) -> None:
        _log(f"Received signal {signum}, shutting down.", log_file=log_file)
        _terminate_child()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    idle_seconds_needed = max(1, int(args.idle_minutes * 60))
    poll = max(5, int(args.poll_seconds))
    consecutive_idle = 0

    while True:
        if child is not None and child.poll() is not None:
            code = child.returncode
            _log(f"Training exited with code {code}. Waiting for idle window again.", log_file=log_file)
            child = None
            consecutive_idle = 0

        if child is None:
            try:
                idle, detail = _is_idle(
                    args.gpu_id,
                    mem_threshold=args.mem_threshold,
                    util_threshold=args.util_threshold,
                )
            except RuntimeError as exc:
                _log(f"GPU query error: {exc}", log_file=log_file)
                time.sleep(poll)
                continue

            if idle:
                consecutive_idle += poll
                if consecutive_idle >= idle_seconds_needed:
                    cmd, mode = _build_train_cmd(args.config)
                    _log(f"Idle for {consecutive_idle}s, starting train ({mode}): {' '.join(cmd)}", log_file=log_file)
                    child = subprocess.Popen(cmd)
                    consecutive_idle = 0
                else:
                    remaining = idle_seconds_needed - consecutive_idle
                    _log(f"Idle {consecutive_idle}/{idle_seconds_needed}s ({detail}); ~{remaining}s left", log_file=log_file)
            else:
                if consecutive_idle > 0:
                    _log(f"GPU busy, reset idle timer. {detail}", log_file=log_file)
                else:
                    _log(f"Waiting... {detail}", log_file=log_file)
                consecutive_idle = 0

        else:
            # Training is running; stop manually via Ctrl+C / kill (runner forwards SIGTERM).
            pass

        time.sleep(poll)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Start/resume finetune_ashare when GPU stays idle long enough."
    )
    parser.add_argument("--config", required=True, help="Training YAML config path")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU index to monitor")
    parser.add_argument(
        "--mem-threshold",
        type=float,
        default=0.5,
        help="Max GPU memory used ratio to count as idle (default 0.5 = 50%%)",
    )
    parser.add_argument(
        "--util-threshold",
        type=float,
        default=20.0,
        help="Max GPU utilization %% to count as idle (default 20)",
    )
    parser.add_argument(
        "--idle-minutes",
        type=float,
        default=30.0,
        help="Required continuous idle duration before starting (default 30)",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=60,
        help="Polling interval in seconds (default 60)",
    )
    parser.add_argument(
        "--log-file",
        default="",
        help="Optional log file path (append)",
    )
    args = parser.parse_args(argv)

    if not 0 < args.mem_threshold < 1:
        parser.error("--mem-threshold must be between 0 and 1")
    if args.idle_minutes <= 0:
        parser.error("--idle-minutes must be positive")

    try:
        run(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)


if __name__ == "__main__":
    main()
