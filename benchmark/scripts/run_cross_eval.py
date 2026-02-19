#!/usr/bin/env python
"""Cross-evaluation job runner for megaseg-benchmark.

Manages launching and tracking cross-evaluation jobs (14 individual DynUNet
models x 14 test sets x 2 dtypes) across multiple GPUs using screen sessions.
Reads experiment_matrix.csv for job definitions.

Usage:
    uv run python scripts/run_cross_eval.py                    # Run all pending jobs
    uv run python scripts/run_cross_eval.py --dry-run          # Show what would be run
    uv run python scripts/run_cross_eval.py --status           # Show job status
    uv run python scripts/run_cross_eval.py --train ACTN1      # Run only ACTN1 model cross-evals
    uv run python scripts/run_cross_eval.py --test ATP2A2      # Run only evals on ATP2A2 test set
    uv run python scripts/run_cross_eval.py --dtype float32    # Run only float32 jobs
    uv run python scripts/run_cross_eval.py --dtype uint8 --train ACTN1  # uint8 ACTN1 cross-evals
    uv run python scripts/run_cross_eval.py --gpus 0,1         # Use only GPUs 0 and 1
    uv run python scripts/run_cross_eval.py --poll-interval 30 # Seconds between status checks
"""

import argparse
import csv
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATRIX_PATH = PROJECT_ROOT / "scripts" / "experiment_matrix.csv"
LOG_DIR = PROJECT_ROOT / "scripts" / "job_logs"

CONFIGS_PATH = PROJECT_ROOT / "configs"
EVAL_SCRIPT = PROJECT_ROOT / "cyto-dl" / "cyto_dl" / "eval.py"

SCREEN_PREFIX = "cross_"
DEFAULT_GPUS = [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def load_matrix(path: Path = MATRIX_PATH) -> list[dict]:
    """Read experiment_matrix.csv and return a list of row dicts."""
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def save_matrix(rows: list[dict], path: Path = MATRIX_PATH) -> None:
    """Write rows back to experiment_matrix.csv, preserving column order."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Output / screen helpers
# ---------------------------------------------------------------------------
def check_output_exists(output_dir: str) -> bool:
    """Return True if output_dir/seg/ contains at least one .tif file."""
    # Resolve relative to project root
    full_path = (
        PROJECT_ROOT / output_dir if not os.path.isabs(output_dir) else Path(output_dir)
    )
    seg_dir = full_path / "seg"
    if not seg_dir.is_dir():
        return False
    return len(glob.glob(str(seg_dir / "*.tif"))) > 0


def get_running_screens() -> set[str]:
    """Return a set of screen session names that match the cross_ prefix."""
    try:
        result = subprocess.run(
            ["screen", "-ls"],
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
    except FileNotFoundError:
        print("WARNING: 'screen' command not found. Cannot detect running sessions.")
        return set()

    sessions: set[str] = set()
    for line in output.splitlines():
        line = line.strip()
        if f".{SCREEN_PREFIX}" in line:
            parts = line.split("\t")[0].strip()
            dot_idx = parts.find(".")
            if dot_idx != -1:
                session_name = parts[dot_idx + 1 :]
                sessions.add(session_name)
    return sessions


def _session_name(row: dict) -> str:
    """Build a deterministic screen session name."""
    return (
        f"{SCREEN_PREFIX}{row['dtype']}_{row['model_struct']}_on_{row['test_struct']}"
    )


# ---------------------------------------------------------------------------
# Job launching
# ---------------------------------------------------------------------------
def launch_job(
    row: dict,
    gpu_id: int,
    dry_run: bool = False,
) -> bool:
    """Launch a single eval job inside a detached screen session.

    Returns True if the job was launched (or would be in dry-run), False on error.
    """
    model_struct = row["model_struct"]
    test_struct = row["test_struct"]
    config_path = row["config_path"]
    output_dir = row["output_dir"]

    session = _session_name(row)
    log_file = LOG_DIR / f"{session}.log"

    # Pre-create the output directory so eval.py / enforce_tags() can write
    # tags.log immediately without a FileNotFoundError.
    full_output = (
        PROJECT_ROOT / output_dir if not os.path.isabs(output_dir) else Path(output_dir)
    )
    full_output.mkdir(parents=True, exist_ok=True)

    eval_cmd = (
        f"cd {PROJECT_ROOT} && "
        f"CUDA_VISIBLE_DEVICES={gpu_id} "
        f"PROJECT_ROOT={PROJECT_ROOT} "
        f"CYTODL_CONFIG_PATH={CONFIGS_PATH} "
        f"uv run python {EVAL_SCRIPT} "
        f"experiment={config_path} "
        f"test_struct={test_struct}"
    )

    if dry_run:
        print(f"  [DRY-RUN] GPU {gpu_id} | {session}")
        print(f"    {eval_cmd}")
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
        eval_cmd,
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
    """Block until at least one GPU from *gpus* is not running a cross_ job."""
    while True:
        running = get_running_screens()
        busy_gpus: set[int] = set()
        for sess in running:
            gpu = gpu_assignments.get(sess)
            if gpu is not None:
                busy_gpus.add(gpu)

        for gpu_id in gpus:
            if gpu_id not in busy_gpus:
                return gpu_id

        print(
            f"  All GPUs busy ({len(busy_gpus)}/{len(gpus)}). "
            f"Waiting {poll_interval}s ..."
        )
        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------
def show_status(rows: list[dict]) -> None:
    """Print a summary of job statuses."""
    total = len(rows)
    completed = sum(1 for r in rows if check_output_exists(r["output_dir"]))
    running_screens = get_running_screens()
    running = sum(
        1
        for r in rows
        if (
            _session_name(r) in running_screens
            and not check_output_exists(r["output_dir"])
        )
    )
    pending = total - completed - running

    print("=" * 60)
    print("Cross-Evaluation Status")
    print("=" * 60)
    print(f"  Total jobs:     {total}")
    print(f"  Completed:      {completed}")
    print(f"  Running:        {running}")
    print(f"  Pending:        {pending}")
    print()

    # Per-dtype breakdown
    dtypes: dict[str, dict[str, int]] = {}
    for r in rows:
        dt = r.get("dtype", "unknown")
        if dt not in dtypes:
            dtypes[dt] = {"completed": 0, "running": 0, "pending": 0}
        done = check_output_exists(r["output_dir"])
        sess = _session_name(r)
        if done:
            dtypes[dt]["completed"] += 1
        elif sess in running_screens:
            dtypes[dt]["running"] += 1
        else:
            dtypes[dt]["pending"] += 1

    print(f"  {'Dtype':<12} {'Done':>6} {'Run':>6} {'Pend':>6}")
    print(f"  {'-' * 12} {'-' * 6} {'-' * 6} {'-' * 6}")
    for dt in sorted(dtypes):
        c = dtypes[dt]
        print(f"  {dt:<12} {c['completed']:>6} {c['running']:>6} {c['pending']:>6}")
    print()

    # Per model_struct breakdown
    model_structs: dict[str, dict[str, int]] = {}
    for r in rows:
        ms = r["model_struct"]
        if ms not in model_structs:
            model_structs[ms] = {"completed": 0, "running": 0, "pending": 0}
        done = check_output_exists(r["output_dir"])
        sess = _session_name(r)
        if done:
            model_structs[ms]["completed"] += 1
        elif sess in running_screens:
            model_structs[ms]["running"] += 1
        else:
            model_structs[ms]["pending"] += 1

    print(f"  {'Model':<12} {'Done':>6} {'Run':>6} {'Pend':>6}")
    print(f"  {'-' * 12} {'-' * 6} {'-' * 6} {'-' * 6}")
    for ms in sorted(model_structs):
        c = model_structs[ms]
        print(f"  {ms:<12} {c['completed']:>6} {c['running']:>6} {c['pending']:>6}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch and track cross-evaluation jobs across GPUs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_cross_eval.py                    # Run all pending\n"
            "  python scripts/run_cross_eval.py --dry-run          # Preview commands\n"
            "  python scripts/run_cross_eval.py --status           # Show job status\n"
            "  python scripts/run_cross_eval.py --train ACTN1      # Only ACTN1 model\n"
            "  python scripts/run_cross_eval.py --test ATP2A2      # Only ATP2A2 test set\n"
            "  python scripts/run_cross_eval.py --dtype float32    # Only float32 jobs\n"
            "  python scripts/run_cross_eval.py --dtype uint8 --train ACTN1  # uint8 ACTN1\n"
            "  python scripts/run_cross_eval.py --gpus 0,1         # Use GPUs 0 and 1\n"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without launching jobs.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current job status and exit.",
    )
    parser.add_argument(
        "--train",
        type=str,
        default=None,
        help="Only run cross-evals for this training structure (model).",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Only run cross-evals on this test structure.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Only run jobs for this output dtype (float32 or uint8). Default: all.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=",".join(str(g) for g in DEFAULT_GPUS),
        help="Comma-separated GPU ids (default: 0,1,2,3).",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between GPU availability checks (default: 30).",
    )
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]

    # Load matrix
    rows = load_matrix()

    # ---- Status mode ----
    if args.status:
        show_status(rows)
        return

    # ---- Filter rows ----
    if args.train:
        rows = [r for r in rows if r["model_struct"] == args.train]
    if args.test:
        rows = [r for r in rows if r["test_struct"] == args.test]
    if args.dtype:
        rows = [r for r in rows if r["dtype"] == args.dtype]

    # Only consider jobs whose output doesn't exist yet
    pending = [r for r in rows if not check_output_exists(r["output_dir"])]

    # Skip jobs that are already running in a screen session
    running_screens = get_running_screens()
    jobs_to_run = [r for r in pending if _session_name(r) not in running_screens]

    if not jobs_to_run:
        print("No pending jobs to launch.")
        show_status(load_matrix())
        return

    print(f"Jobs to launch: {len(jobs_to_run)}  |  GPUs: {gpus}")
    if args.dry_run:
        print("--- DRY RUN (no jobs will be launched) ---\n")

    # Ensure log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    gpu_assignments: dict[str, int] = {}

    launched = 0
    for i, row in enumerate(jobs_to_run):
        session = _session_name(row)

        # Skip if output already appeared (race condition guard)
        if check_output_exists(row["output_dir"]):
            continue

        if not args.dry_run:
            gpu_id = wait_for_gpu_slot(gpus, gpu_assignments, args.poll_interval)
        else:
            gpu_id = gpus[i % len(gpus)]

        ok = launch_job(
            row=row,
            gpu_id=gpu_id,
            dry_run=args.dry_run,
        )

        if ok:
            launched += 1
            if not args.dry_run:
                gpu_assignments[session] = gpu_id
                time.sleep(2)

            if launched % 10 == 0:
                print(f"  ... launched {launched}/{len(jobs_to_run)} jobs")

    print(f"\nLaunched {launched} job(s).")

    if not args.dry_run and launched > 0:
        print("\nMonitoring:")
        print(f"  python scripts/run_cross_eval.py --status")
        print("  screen -ls | grep cross_")
        print(f"\nLogs: {LOG_DIR}/")


if __name__ == "__main__":
    main()
