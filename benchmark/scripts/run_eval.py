#!/usr/bin/env python
"""General-purpose batch eval runner for megaseg-benchmark.

Generates and manages ALL evaluation jobs (individual, individual_test_new,
megaseg, cross_eval) across multiple GPUs using screen sessions.  Job
definitions are derived declaratively from known constants — no CSV or
filesystem scanning required.

Job counts per dtype:
    individual:          14  (14 structures)
    individual_test_new: 14  (14 structures)
    megaseg:             60  (4 backbones × 15: 14 structures + test_new)
    cross_eval:         196  (14 model structures × 14 test structures)
    ────────────────────────
    Total:              284 per dtype, 568 for both

Usage:
    # Run all uint8 evals
    python scripts/run_eval.py --dtype uint8 --gpus 0,1,2,3

    # Run only individual uint8
    python scripts/run_eval.py --dtype uint8 --type individual

    # Run megaseg dynunet only
    python scripts/run_eval.py --dtype uint8 --type megaseg --backbone dynunet

    # Filter by structure
    python scripts/run_eval.py --dtype uint8 --type individual --struct ACTN1

    # Show completion status
    python scripts/run_eval.py --status --dtype uint8

    # Dry-run preview
    python scripts/run_eval.py --dry-run --dtype uint8 --type individual

    # Everything
    python scripts/run_eval.py --dtype all --type all

    # Dry-run with specific GPUs
    python scripts/run_eval.py --dry-run --dtype all --gpus 0,1

    # Pack 8 jobs per GPU (A100-80GB)
    python scripts/run_eval.py --dtype uint8 --type megaseg --gpus 0,1,2,3 --jobs-per-gpu 8
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
CONDA_ENV = "acm-eval"
# Resolve conda base so screen sessions can source conda.sh in non-login shells.
_conda_base = subprocess.run(
    ["conda", "info", "--base"], capture_output=True, text=True, check=True
).stdout.strip()
CONDA_SH = Path(_conda_base) / "etc" / "profile.d" / "conda.sh"
EVAL_SCRIPT = PROJECT_ROOT / "cyto-dl" / "cyto_dl" / "eval.py"

SCREEN_PREFIX = "eval_"
DEFAULT_GPUS = [0, 1, 2, 3]

STRUCTURES = [
    "ACTN1",
    "ATP2A2",
    "CTNNB1",
    "DSP",
    "GJA1",
    "MYH10",
    "NPM1",
    "NUP153",
    "RAB5A",
    "SEC61B",
    "SON",
    "ST6GAL1",
    "TJP1",
    "TOMM20",
]
BACKBONES = ["dynunet", "attention_unet", "swin_unetr", "unet"]
DTYPES = ["float32", "uint8"]
EVAL_TYPES = ["individual", "individual_test_new", "megaseg", "cross_eval"]

# Expected .tif file counts in output_dir/seg/
EXPECTED_FILES_STANDARD = 67  # standard test set
EXPECTED_FILES_TEST_NEW = 49  # test_new set


# ---------------------------------------------------------------------------
# Job definition
# ---------------------------------------------------------------------------
class EvalJob:
    """A single evaluation job with all metadata needed to launch and check it."""

    __slots__ = (
        "dtype",
        "eval_type",
        "config_path",
        "output_dir",
        "expected_files",
        "extra_args",
        "label",
        "backbone",
    )

    def __init__(
        self,
        dtype: str,
        eval_type: str,
        config_path: str,
        output_dir: str,
        expected_files: int,
        extra_args: Optional[dict[str, str]] = None,
        label: str = "",
        backbone: Optional[str] = None,
    ) -> None:
        self.dtype = dtype
        self.eval_type = eval_type
        self.config_path = config_path
        self.output_dir = output_dir
        self.expected_files = expected_files
        self.extra_args = extra_args or {}
        self.label = label
        self.backbone = backbone

    @property
    def session_name(self) -> str:
        """Build a deterministic screen session name."""
        # Sanitise the label to be screen-friendly (no slashes)
        safe = self.label.replace("/", "_").replace(" ", "_")
        return f"{SCREEN_PREFIX}{self.dtype}_{safe}"


# ---------------------------------------------------------------------------
# Job generation (declarative — no filesystem scanning)
# ---------------------------------------------------------------------------
def generate_jobs(
    dtypes: list[str],
    eval_types: list[str],
    backbone_filter: Optional[str] = None,
    struct_filter: Optional[str] = None,
) -> list[EvalJob]:
    """Generate the full list of eval jobs from known constants."""
    jobs: list[EvalJob] = []

    for dtype in dtypes:
        if "individual" in eval_types:
            for struct in STRUCTURES:
                if struct_filter and struct != struct_filter:
                    continue
                jobs.append(
                    EvalJob(
                        dtype=dtype,
                        eval_type="individual",
                        config_path=f"eval/{dtype}/individual/{struct}",
                        output_dir=f"outputs/eval/{dtype}/individual/ms13-{struct}/",
                        expected_files=EXPECTED_FILES_STANDARD,
                        label=f"individual_{struct}",
                    )
                )

        if "individual_test_new" in eval_types:
            for struct in STRUCTURES:
                if struct_filter and struct != struct_filter:
                    continue
                jobs.append(
                    EvalJob(
                        dtype=dtype,
                        eval_type="individual_test_new",
                        config_path=f"eval/{dtype}/individual_test_new/{struct}",
                        output_dir=f"outputs/eval/{dtype}/individual_test_new/ms13-{struct}/",
                        expected_files=EXPECTED_FILES_TEST_NEW,
                        label=f"individual_test_new_{struct}",
                    )
                )

        if "megaseg" in eval_types:
            for backbone in BACKBONES:
                if backbone_filter and backbone != backbone_filter:
                    continue

                # Per-structure megaseg evals
                for struct in STRUCTURES:
                    if struct_filter and struct != struct_filter:
                        continue
                    jobs.append(
                        EvalJob(
                            dtype=dtype,
                            eval_type="megaseg",
                            config_path=f"eval/{dtype}/megaseg/{backbone}/{struct}",
                            output_dir=f"outputs/eval/{dtype}/megaseg/{backbone}/{struct}/",
                            expected_files=EXPECTED_FILES_STANDARD,
                            label=f"megaseg_{backbone}_{struct}",
                            backbone=backbone,
                        )
                    )

                # test_new megaseg eval (not filtered by struct_filter)
                if not struct_filter:
                    jobs.append(
                        EvalJob(
                            dtype=dtype,
                            eval_type="megaseg",
                            config_path=f"eval/{dtype}/megaseg/{backbone}/test_new",
                            output_dir=f"outputs/eval/{dtype}/megaseg/{backbone}/test_new/",
                            expected_files=EXPECTED_FILES_TEST_NEW,
                            label=f"megaseg_{backbone}_test_new",
                            backbone=backbone,
                        )
                    )

        if "cross_eval" in eval_types:
            for model_struct in STRUCTURES:
                if struct_filter and model_struct != struct_filter:
                    continue
                for test_struct in STRUCTURES:
                    jobs.append(
                        EvalJob(
                            dtype=dtype,
                            eval_type="cross_eval",
                            config_path=f"eval/{dtype}/cross_eval/{model_struct}_model/cross_eval",
                            output_dir=f"outputs/eval/{dtype}/cross_eval/ms13-{model_struct}/tested_on_{test_struct}/",
                            expected_files=EXPECTED_FILES_STANDARD,
                            extra_args={"test_struct": test_struct},
                            label=f"cross_eval_{model_struct}_on_{test_struct}",
                        )
                    )

    return jobs


# ---------------------------------------------------------------------------
# Output / screen helpers
# ---------------------------------------------------------------------------
def check_output_complete(job: EvalJob) -> bool:
    """Return True if the job's output_dir/seg/ contains the expected .tif count."""
    full_path = (
        PROJECT_ROOT / job.output_dir
        if not os.path.isabs(job.output_dir)
        else Path(job.output_dir)
    )
    seg_dir = full_path / "seg"
    if not seg_dir.is_dir():
        return False
    return len(glob.glob(str(seg_dir / "*.tif"))) >= job.expected_files


def get_running_screens() -> set[str]:
    """Return a set of screen session names that match the eval_ prefix."""
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


# ---------------------------------------------------------------------------
# Job launching
# ---------------------------------------------------------------------------
def launch_job(
    job: EvalJob,
    gpu_id: int,
    dry_run: bool = False,
) -> bool:
    """Launch a single eval job inside a detached screen session.

    Returns True if the job was launched (or would be in dry-run), False on error.
    """
    session = job.session_name
    log_file = LOG_DIR / f"{session}.log"

    # Pre-create the output directory so eval.py doesn't hit FileNotFoundError.
    full_output = (
        PROJECT_ROOT / job.output_dir
        if not os.path.isabs(job.output_dir)
        else Path(job.output_dir)
    )
    full_output.mkdir(parents=True, exist_ok=True)

    # Build eval command
    extra_args_str = " ".join(f"{k}={v}" for k, v in job.extra_args.items())
    eval_cmd = (
        f"source {CONDA_SH} && conda activate {CONDA_ENV} && "
        f"cd {PROJECT_ROOT} && "
        f"CUDA_VISIBLE_DEVICES={gpu_id} "
        f"PROJECT_ROOT={PROJECT_ROOT} "
        f"CYTODL_CONFIG_PATH={CONFIGS_PATH} "
        f"python {EVAL_SCRIPT} "
        f"experiment={job.config_path}"
    )
    if extra_args_str:
        eval_cmd += f" {extra_args_str}"

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
    jobs_per_gpu: int = 1,
) -> int:
    """Block until a GPU from *gpus* has fewer than *jobs_per_gpu* running."""
    while True:
        running = get_running_screens()
        gpu_load: dict[int, int] = {g: 0 for g in gpus}
        for sess in running:
            gpu = gpu_assignments.get(sess)
            if gpu is not None and gpu in gpu_load:
                gpu_load[gpu] += 1

        # Pick the least-loaded GPU that has capacity
        for gpu_id in sorted(gpu_load, key=lambda g: gpu_load[g]):
            if gpu_load[gpu_id] < jobs_per_gpu:
                return gpu_id

        print(
            f"  All GPU slots full ({sum(gpu_load.values())}/{len(gpus) * jobs_per_gpu}). "
            f"Waiting {poll_interval}s ..."
        )
        time.sleep(poll_interval)


def infer_gpu_assignments_from_logs() -> dict[str, int]:
    """Scan log files for already-running eval_ screens to recover GPU IDs.

    When the orchestrator is restarted, ``gpu_assignments`` starts empty.  This
    function reads each log file under ``LOG_DIR`` whose session name matches a
    currently-running screen and extracts the ``CUDA_VISIBLE_DEVICES`` value so
    we can pre-populate the assignment map.
    """
    running = get_running_screens()
    assignments: dict[str, int] = {}
    if not LOG_DIR.is_dir():
        return assignments

    for log_path in LOG_DIR.glob("*.log"):
        session_name = log_path.stem  # e.g. eval_uint8_individual_ATP2A2
        if session_name not in running:
            continue
        try:
            text = log_path.read_text(errors="replace")
        except OSError:
            continue
        # Match lines like: CUDA_VISIBLE_DEVICES: [0]  or  CUDA_VISIBLE_DEVICES=0
        m = re.search(r"CUDA_VISIBLE_DEVICES[=:\s]+\[?(\d+)", text)
        if m:
            assignments[session_name] = int(m.group(1))
    return assignments


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------
def show_status(jobs: list[EvalJob]) -> None:
    """Print a summary of job statuses."""
    running_screens = get_running_screens()

    def _classify(job: EvalJob) -> str:
        if check_output_complete(job):
            return "completed"
        if job.session_name in running_screens:
            return "running"
        return "pending"

    total = len(jobs)
    statuses = [_classify(j) for j in jobs]
    completed = statuses.count("completed")
    running = statuses.count("running")
    pending = statuses.count("pending")

    print("=" * 65)
    print("Evaluation Status")
    print("=" * 65)
    print(f"  Total jobs:     {total}")
    print(f"  Completed:      {completed}")
    print(f"  Running:        {running}")
    print(f"  Pending:        {pending}")
    print()

    # Per-dtype breakdown
    dtype_stats: dict[str, dict[str, int]] = {}
    for job, status in zip(jobs, statuses):
        dt = job.dtype
        if dt not in dtype_stats:
            dtype_stats[dt] = {"completed": 0, "running": 0, "pending": 0}
        dtype_stats[dt][status] += 1

    print(f"  {'Dtype':<12} {'Done':>6} {'Run':>6} {'Pend':>6}")
    print(f"  {'-' * 12} {'-' * 6} {'-' * 6} {'-' * 6}")
    for dt in sorted(dtype_stats):
        c = dtype_stats[dt]
        print(f"  {dt:<12} {c['completed']:>6} {c['running']:>6} {c['pending']:>6}")
    print()

    # Per eval_type breakdown
    type_stats: dict[str, dict[str, int]] = {}
    for job, status in zip(jobs, statuses):
        et = job.eval_type
        if et not in type_stats:
            type_stats[et] = {"completed": 0, "running": 0, "pending": 0}
        type_stats[et][status] += 1

    print(f"  {'Eval Type':<22} {'Done':>6} {'Run':>6} {'Pend':>6}")
    print(f"  {'-' * 22} {'-' * 6} {'-' * 6} {'-' * 6}")
    for et in EVAL_TYPES:
        if et in type_stats:
            c = type_stats[et]
            print(f"  {et:<22} {c['completed']:>6} {c['running']:>6} {c['pending']:>6}")
    print()

    # Per backbone breakdown (megaseg only)
    megaseg_jobs = [(j, s) for j, s in zip(jobs, statuses) if j.eval_type == "megaseg"]
    if megaseg_jobs:
        backbone_stats: dict[str, dict[str, int]] = {}
        for job, status in megaseg_jobs:
            backbone = job.backbone or "unknown"
            if backbone not in backbone_stats:
                backbone_stats[backbone] = {"completed": 0, "running": 0, "pending": 0}
            backbone_stats[backbone][status] += 1

        print(f"  {'Backbone':<18} {'Done':>6} {'Run':>6} {'Pend':>6}")
        print(f"  {'-' * 18} {'-' * 6} {'-' * 6} {'-' * 6}")
        for bb in BACKBONES:
            if bb in backbone_stats:
                c = backbone_stats[bb]
                print(
                    f"  {bb:<18} {c['completed']:>6} {c['running']:>6} {c['pending']:>6}"
                )
        print()

    print("=" * 65)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch and track evaluation jobs across GPUs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_eval.py --dtype uint8 --gpus 0,1,2,3\n"
            "  python scripts/run_eval.py --dtype uint8 --type individual\n"
            "  python scripts/run_eval.py --dtype uint8 --type megaseg --backbone dynunet\n"
            "  python scripts/run_eval.py --status --dtype uint8\n"
            "  python scripts/run_eval.py --dry-run --dtype uint8 --type individual\n"
            "  python scripts/run_eval.py --dtype all --type all\n"
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        required=True,
        help="Output dtype: float32, uint8, or all.",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="all",
        help=(
            "Eval type: individual, individual_test_new, megaseg, cross_eval, "
            "or all (default: all)."
        ),
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Filter megaseg jobs by backbone name (only applies to megaseg type).",
    )
    parser.add_argument(
        "--struct",
        type=str,
        default=None,
        help="Filter by structure name (applies to all types).",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=",".join(str(g) for g in DEFAULT_GPUS),
        help="Comma-separated GPU ids (default: 0,1,2,3).",
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
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between GPU availability checks (default: 30).",
    )
    parser.add_argument(
        "--jobs-per-gpu",
        type=int,
        default=1,
        help="Max concurrent jobs per GPU (default: 1). A100-80GB can handle ~8.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]

    # Resolve dtype list
    if args.dtype == "all":
        dtypes = DTYPES
    else:
        if args.dtype not in DTYPES:
            print(f"ERROR: --dtype must be one of {DTYPES} or 'all'", file=sys.stderr)
            sys.exit(1)
        dtypes = [args.dtype]

    # Resolve eval type list
    if args.type == "all":
        eval_types = EVAL_TYPES
    else:
        if args.type not in EVAL_TYPES:
            print(
                f"ERROR: --type must be one of {EVAL_TYPES} or 'all'",
                file=sys.stderr,
            )
            sys.exit(1)
        eval_types = [args.type]

    # Validate backbone filter
    if args.backbone and args.backbone not in BACKBONES:
        print(f"ERROR: --backbone must be one of {BACKBONES}", file=sys.stderr)
        sys.exit(1)

    # Validate struct filter
    if args.struct and args.struct not in STRUCTURES:
        print(f"ERROR: --struct must be one of {STRUCTURES}", file=sys.stderr)
        sys.exit(1)

    # Generate jobs
    jobs = generate_jobs(
        dtypes=dtypes,
        eval_types=eval_types,
        backbone_filter=args.backbone,
        struct_filter=args.struct,
    )

    # ---- Status mode ----
    if args.status:
        show_status(jobs)
        return

    # Only consider jobs whose output isn't already complete
    pending = [j for j in jobs if not check_output_complete(j)]

    # Skip jobs that are already running in a screen session
    running_screens = get_running_screens()
    jobs_to_run = [j for j in pending if j.session_name not in running_screens]

    if not jobs_to_run:
        print("No pending jobs to launch.")
        show_status(jobs)
        return

    print(
        f"Jobs to launch: {len(jobs_to_run)}  |  GPUs: {gpus}  |  Jobs/GPU: {args.jobs_per_gpu}"
    )
    if args.dry_run:
        print("--- DRY RUN (no jobs will be launched) ---\n")

    # Ensure log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-populate GPU assignments from already-running screen sessions so we
    # don't collide with jobs that were launched by a prior orchestrator run.
    gpu_assignments: dict[str, int] = infer_gpu_assignments_from_logs()
    if gpu_assignments:
        print(
            f"  Recovered {len(gpu_assignments)} GPU assignment(s) from running sessions:"
        )
        for sess, gpu in sorted(gpu_assignments.items()):
            print(f"    {sess} -> GPU {gpu}")
        print()

    launched = 0
    for i, job in enumerate(jobs_to_run):
        session = job.session_name

        # Skip if output already appeared (race condition guard)
        if check_output_complete(job):
            continue

        if not args.dry_run:
            gpu_id = wait_for_gpu_slot(
                gpus, gpu_assignments, args.poll_interval, args.jobs_per_gpu
            )
        else:
            gpu_id = gpus[i % len(gpus)]

        ok = launch_job(
            job=job,
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

    print(f"\nTotal jobs generated: {len(jobs)}")
    print(f"Launched {launched} job(s).")

    if not args.dry_run and launched > 0:
        print("\nMonitoring:")
        print(f"  python scripts/run_eval.py --status --dtype {args.dtype}")
        print("  screen -ls | grep eval_")
        print(f"\nLogs: {LOG_DIR}/")


if __name__ == "__main__":
    main()
