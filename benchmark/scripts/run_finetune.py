#!/usr/bin/env python
"""Batch fine-tuning runner for LMNB1 experiments.

Manages 16 training jobs across multiple GPUs using screen sessions.
Each job fine-tunes the MegaSeg DynUNet on LMNB1 data with a specific
data fraction and epoch configuration.

Usage:
    # Run all 16 experiments
    uv run python scripts/run_finetune.py --gpus 0,1,2,3

    # Run specific experiments
    uv run python scripts/run_finetune.py --gpus 0 --experiments data100_epochs10,data50_epochs50

    # Show status
    uv run python scripts/run_finetune.py --status

    # Dry-run preview
    uv run python scripts/run_finetune.py --dry-run --gpus 0,1,2,3
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "scripts" / "job_logs"

CONFIGS_PATH = PROJECT_ROOT / "configs"
TRAIN_SCRIPT = PROJECT_ROOT / "cyto-dl" / "cyto_dl" / "train.py"

SCREEN_PREFIX = "ft_"
DEFAULT_GPUS = [0, 1, 2, 3]

# All 16 experiment labels
DATA_FRACTIONS = [100, 50, 20, 10]
EPOCH_CONFIGS = ["epochs10", "epochs50", "epochs100", "earlystop"]
ALL_EXPERIMENTS = [f"data{pct}_{ec}" for pct in DATA_FRACTIONS for ec in EPOCH_CONFIGS]

# Expected .tif count in output seg/ dir (67 test FOVs)
EXPECTED_TEST_FILES = 67


# ---------------------------------------------------------------------------
# Job definition
# ---------------------------------------------------------------------------
class FinetuneJob:
    """A single fine-tuning job."""

    __slots__ = ("label", "config_path", "output_dir")

    def __init__(self, label: str) -> None:
        self.label = label
        self.config_path = f"finetune/lmnb1/{label}"
        self.output_dir = f"outputs/finetune/lmnb1/{label}"

    @property
    def session_name(self) -> str:
        return f"{SCREEN_PREFIX}{self.label}"


def generate_jobs(experiments: Optional[list[str]] = None) -> list[FinetuneJob]:
    """Generate fine-tuning jobs, optionally filtered."""
    labels = experiments if experiments else ALL_EXPERIMENTS
    return [FinetuneJob(label) for label in labels]


# ---------------------------------------------------------------------------
# Output / screen helpers
# ---------------------------------------------------------------------------
def check_output_complete(job: FinetuneJob) -> bool:
    """Return True if the job's test output has expected .tif count."""
    seg_dir = PROJECT_ROOT / job.output_dir / "seg"
    if not seg_dir.is_dir():
        return False
    return len(glob.glob(str(seg_dir / "*.tif*"))) >= EXPECTED_TEST_FILES


def check_checkpoint_exists(job: FinetuneJob) -> bool:
    """Return True if a checkpoint was saved (training started/completed)."""
    ckpt_dir = PROJECT_ROOT / job.output_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return False
    return len(list(ckpt_dir.glob("*.ckpt"))) > 0


def get_running_screens() -> set[str]:
    """Return screen session names matching the finetune prefix."""
    try:
        result = subprocess.run(
            ["screen", "-ls"],
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
    except FileNotFoundError:
        print("WARNING: 'screen' not found.")
        return set()

    sessions: set[str] = set()
    for line in output.splitlines():
        line = line.strip()
        if f".{SCREEN_PREFIX}" in line:
            parts = line.split("\t")[0].strip()
            dot_idx = parts.find(".")
            if dot_idx != -1:
                sessions.add(parts[dot_idx + 1 :])
    return sessions


# ---------------------------------------------------------------------------
# Job launching
# ---------------------------------------------------------------------------
def launch_job(job: FinetuneJob, gpu_id: int, dry_run: bool = False) -> bool:
    """Launch a training job in a detached screen session."""
    session = job.session_name
    log_file = LOG_DIR / f"{session}.log"

    full_output = PROJECT_ROOT / job.output_dir
    full_output.mkdir(parents=True, exist_ok=True)

    train_cmd = (
        f"cd {PROJECT_ROOT} && "
        f"CUDA_VISIBLE_DEVICES={gpu_id} "
        f"PROJECT_ROOT={PROJECT_ROOT} "
        f"CYTODL_CONFIG_PATH={CONFIGS_PATH} "
        f"uv run python {TRAIN_SCRIPT} "
        f"experiment={job.config_path}"
    )

    if dry_run:
        print(f"  [DRY-RUN] GPU {gpu_id} | {session}")
        print(f"    {train_cmd}")
        return True

    screen_cmd = [
        "screen",
        "-dmS",
        session,
        "-L",
        "-Logfile",
        str(log_file),
        "bash",
        "-c",
        train_cmd,
    ]

    try:
        subprocess.run(screen_cmd, check=True)
        print(f"  Launched GPU {gpu_id} | {session}")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"  ERROR launching {session}: {exc}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# GPU slot management
# ---------------------------------------------------------------------------
def wait_for_gpu_slot(
    gpus: list[int],
    gpu_assignments: dict[str, int],
    poll_interval: int,
) -> int:
    """Block until a GPU has no running finetune jobs."""
    while True:
        running = get_running_screens()
        gpu_load: dict[int, int] = {g: 0 for g in gpus}
        for sess in running:
            gpu = gpu_assignments.get(sess)
            if gpu is not None and gpu in gpu_load:
                gpu_load[gpu] += 1

        for gpu_id in sorted(gpu_load, key=lambda g: gpu_load[g]):
            if gpu_load[gpu_id] < 1:
                return gpu_id

        print(
            f"  All GPUs busy ({sum(gpu_load.values())}/{len(gpus)}). "
            f"Waiting {poll_interval}s ..."
        )
        time.sleep(poll_interval)


def infer_gpu_assignments_from_logs() -> dict[str, int]:
    """Recover GPU assignments from log files of running sessions."""
    running = get_running_screens()
    assignments: dict[str, int] = {}
    if not LOG_DIR.is_dir():
        return assignments

    for log_path in LOG_DIR.glob("*.log"):
        session_name = log_path.stem
        if session_name not in running:
            continue
        try:
            text = log_path.read_text(errors="replace")
        except OSError:
            continue
        m = re.search(r"CUDA_VISIBLE_DEVICES[=:\s]+\[?(\d+)", text)
        if m:
            assignments[session_name] = int(m.group(1))
    return assignments


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------
def show_status(jobs: list[FinetuneJob]) -> None:
    """Print job statuses."""
    running_screens = get_running_screens()

    print("=" * 65)
    print("Fine-tuning Status (LMNB1)")
    print("=" * 65)

    completed = 0
    running = 0
    has_ckpt = 0
    pending = 0

    for job in jobs:
        if check_output_complete(job):
            status = "DONE"
            completed += 1
        elif job.session_name in running_screens:
            status = "RUNNING"
            running += 1
        elif check_checkpoint_exists(job):
            status = "CKPT (no test output)"
            has_ckpt += 1
        else:
            status = "PENDING"
            pending += 1

        print(f"  {job.label:<28} {status}")

    print(
        f"\n  Total: {len(jobs)}  |  Done: {completed}  |  "
        f"Running: {running}  |  Ckpt: {has_ckpt}  |  Pending: {pending}"
    )
    print("=" * 65)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch and track LMNB1 fine-tuning jobs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=",".join(str(g) for g in DEFAULT_GPUS),
        help="Comma-separated GPU ids (default: 0,1,2,3).",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help="Comma-separated experiment labels (default: all 16).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between GPU checks (default: 60).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]

    exp_filter = None
    if args.experiments:
        exp_filter = [e.strip() for e in args.experiments.split(",")]
        for e in exp_filter:
            if e not in ALL_EXPERIMENTS:
                print(f"ERROR: Unknown experiment '{e}'", file=sys.stderr)
                print(f"Valid: {ALL_EXPERIMENTS}", file=sys.stderr)
                sys.exit(1)

    jobs = generate_jobs(exp_filter)

    if args.status:
        show_status(jobs)
        return

    pending = [j for j in jobs if not check_output_complete(j)]
    running_screens = get_running_screens()
    jobs_to_run = [j for j in pending if j.session_name not in running_screens]

    if not jobs_to_run:
        print("No pending jobs to launch.")
        show_status(jobs)
        return

    print(f"Jobs to launch: {len(jobs_to_run)}  |  GPUs: {gpus}")
    if args.dry_run:
        print("--- DRY RUN ---\n")

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    gpu_assignments = infer_gpu_assignments_from_logs()
    if gpu_assignments:
        print(f"  Recovered {len(gpu_assignments)} GPU assignment(s):")
        for sess, gpu in sorted(gpu_assignments.items()):
            print(f"    {sess} -> GPU {gpu}")
        print()

    launched = 0
    for i, job in enumerate(jobs_to_run):
        if check_output_complete(job):
            continue

        if not args.dry_run:
            gpu_id = wait_for_gpu_slot(gpus, gpu_assignments, args.poll_interval)
        else:
            gpu_id = gpus[i % len(gpus)]

        ok = launch_job(job, gpu_id, dry_run=args.dry_run)
        if ok:
            launched += 1
            if not args.dry_run:
                gpu_assignments[job.session_name] = gpu_id
                time.sleep(2)

    print(f"\nLaunched {launched} job(s).")
    if not args.dry_run and launched > 0:
        print("\nMonitor:")
        print("  uv run python scripts/run_finetune.py --status")
        print("  screen -ls | grep ft_")
        print(f"  Logs: {LOG_DIR}/")


if __name__ == "__main__":
    main()
