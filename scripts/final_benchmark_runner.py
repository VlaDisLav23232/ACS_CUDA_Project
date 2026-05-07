#!/usr/bin/env python3
"""Final benchmark runner for ACS CUDA project.

This script is intentionally conservative:
- appends raw rows after every run
- keeps stdout/stderr logs for every command
- continues after failures
- writes metadata and summary stats
- shows tqdm progress and ETA when tqdm is installed

Example:
    python3 scripts/final_benchmark_runner.py --profile smoke
    python3 scripts/final_benchmark_runner.py --profile calibration --repeats 1
    python3 scripts/final_benchmark_runner.py --profile validated --repeats 5
    python3 scripts/final_benchmark_runner.py --profile calibration --out-dir results/final/overnight --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

try:
    from tqdm import tqdm
except ImportError:  # simple fallback, still useful on a bare machine
    tqdm = None


ROOT = Path(__file__).resolve().parents[1]
HEAT_DIR = ROOT / "cuda-heat-equation"
WAVE_DIR = ROOT / "cuda-wave-equation"
DEFAULT_OUT = ROOT / "results" / "final"
STOP_REQUESTED = False


RAW_COLUMNS = [
    "run_id",
    "timestamp",
    "profile",
    "project",
    "equation",
    "git_commit",
    "gpu_name",
    "compute_cap",
    "driver_version",
    "cuda_arch",
    "variant_request",
    "variant",
    "dim",
    "grid_size",
    "reach",
    "timesteps",
    "repeat",
    "warmup_runs",
    "elapsed_ms",
    "megapoints_per_sec",
    "bandwidth_gbs",
    "max_abs_error",
    "l2_error",
    "memory_bytes",
    "status",
    "exit_code",
    "duration_wall_sec",
    "stdout_log",
    "stderr_log",
]


SUMMARY_COLUMNS = [
    "profile",
    "project",
    "equation",
    "variant",
    "dim",
    "grid_size",
    "reach",
    "timesteps",
    "runs_ok",
    "runs_failed",
    "elapsed_ms_mean",
    "elapsed_ms_median",
    "elapsed_ms_std",
    "elapsed_ms_min",
    "elapsed_ms_max",
    "mpoints_mean",
    "mpoints_median",
    "mpoints_std",
    "mpoints_min",
    "mpoints_max",
    "max_abs_error_mean",
    "l2_error_mean",
]


@dataclass(frozen=True)
class BenchCase:
    project: str
    dim: int
    grid_size: int
    reach: int
    timesteps: int
    variant: str

    @property
    def equation(self) -> str:
        return self.project


def request_stop(signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"\nreceived signal {signum}; will stop after the current case and write summaries", flush=True)


def run_cmd(cmd: list[str], cwd: Path, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def command_text(cmd: Iterable[str]) -> str:
    return " ".join(str(part) for part in cmd)


def get_git_commit() -> str:
    proc = run_cmd(["git", "rev-parse", "--short", "HEAD"], ROOT)
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def get_gpu_info() -> dict[str, str]:
    proc = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,driver_version",
            "--format=csv,noheader",
        ],
        ROOT,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return {"gpu_name": "unknown", "compute_cap": "unknown", "driver_version": "unknown"}

    first = proc.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    while len(parts) < 3:
        parts.append("unknown")
    return {"gpu_name": parts[0], "compute_cap": parts[1], "driver_version": parts[2]}


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def append_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def read_result_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def maybe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def derived_mpoints(case: BenchCase, result_row: dict[str, str]) -> str:
    elapsed_ms = maybe_float(result_row.get("elapsed_ms"), 0.0)
    if elapsed_ms <= 0.0:
        return result_row.get("megapoints_per_sec", "")

    points = float(case.grid_size ** case.dim) * case.timesteps
    calculated = points / (elapsed_ms / 1000.0) / 1e6
    existing = maybe_float(result_row.get("megapoints_per_sec"), 0.0)

    if existing > 0.001 and existing < 1.0e9:
        return result_row.get("megapoints_per_sec", "")
    return f"{calculated:.6g}"


def project_dir(project: str) -> Path:
    if project == "heat":
        return HEAT_DIR
    if project == "wave":
        return WAVE_DIR
    raise ValueError(f"unknown project: {project}")


def run_id_for(case: BenchCase, repeat: int) -> str:
    return f"{case.project}_{case.dim}d_n{case.grid_size}_r{case.reach}_t{case.timesteps}_{case.variant}_rep{repeat}"


def completed_run_ids(raw_csv: Path) -> set[str]:
    if not raw_csv.exists():
        return set()
    with raw_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    latest: dict[str, str] = {}
    for row in rows:
        run_id = row.get("run_id", "")
        if run_id:
            latest[run_id] = row.get("status", "")
    return {run_id for run_id, status in latest.items() if status == "ok"}


def binary_path(project: str, build_name: str) -> Path:
    exe = "heat_stencil" if project == "heat" else "wave_stencil"
    return project_dir(project) / build_name / exe


def build_project(project: str, build_name: str, cuda_arch: str) -> None:
    src = project_dir(project)
    build = src / build_name
    cfg = run_cmd(
        [
            "cmake",
            "-B",
            build_name,
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}",
        ],
        src,
        timeout=180,
    )
    if cfg.returncode != 0:
        raise RuntimeError(f"configure failed for {project}\n{cfg.stdout}\n{cfg.stderr}")

    build_proc = run_cmd(["cmake", "--build", build_name, f"-j{os.cpu_count() or 4}"], src, timeout=600)
    if build_proc.returncode != 0:
        raise RuntimeError(f"build failed for {project}\n{build_proc.stdout}\n{build_proc.stderr}")

    if not binary_path(project, build_name).exists():
        raise RuntimeError(f"missing binary after build: {binary_path(project, build_name)}")


def build_projects(projects: set[str], build_name: str, cuda_arch: str) -> None:
    for project in sorted(projects):
        print(f"[build] {project} -> {build_name}, sm_{cuda_arch}", flush=True)
        build_project(project, build_name, cuda_arch)


def heat_2d_variants(include_experimental: bool) -> list[str]:
    variants = ["fp32", "fp16", "kahan", "tensor"]
    if include_experimental:
        variants += ["cfp16", "cfp16_kahan", "cfp16_kahan_tiled"]
    return variants


def heat_3d_variants(include_experimental: bool) -> list[str]:
    variants = ["fp32", "fp16", "kahan", "fp32_25d", "fp32_25d_zreg", "fp16_25d_zreg", "fp16_25d_zreg_async", "kahan_25d", "kahan_25d_zreg", "kahan_25d_zreg_async", "kahan_25d_async"]
    if include_experimental:
        variants += ["cfp16_kahan_3d_tiled"]
    return variants


def wave_2d_variants() -> list[str]:
    return ["fp32", "fp16", "kahan", "tensor"]


def wave_3d_variants() -> list[str]:
    return ["fp32", "fp16", "kahan", "fp32_25d", "kahan_25d", "kahan_25d_async"]


def make_cases(profile: str, include_wave: bool, include_experimental: bool) -> list[BenchCase]:
    cases: list[BenchCase] = []

    if profile == "smoke":
        for variant in heat_2d_variants(include_experimental):
            cases.append(BenchCase("heat", 2, 64, 1, 10, variant))
        for variant in heat_3d_variants(include_experimental):
            cases.append(BenchCase("heat", 3, 32, 1, 10, variant))
        if include_wave:
            for variant in wave_2d_variants():
                cases.append(BenchCase("wave", 2, 64, 1, 10, variant))
            for variant in wave_3d_variants():
                cases.append(BenchCase("wave", 3, 32, 1, 10, variant))
        return cases

    if profile == "calibration":
        for reach in [1, 4, 8]:
            for grid_size in [64, 96, 128]:
                for variant in heat_3d_variants(include_experimental):
                    cases.append(BenchCase("heat", 3, grid_size, reach, 50, variant))
        for reach in [1, 4, 8]:
            for grid_size in [128, 256, 512]:
                for variant in heat_2d_variants(include_experimental):
                    cases.append(BenchCase("heat", 2, grid_size, reach, 100, variant))
        if include_wave:
            for reach in [1, 4]:
                for grid_size in [64, 128]:
                    for variant in wave_3d_variants():
                        cases.append(BenchCase("wave", 3, grid_size, reach, 50, variant))
        return cases

    if profile == "validated":
        for reach in [1, 4, 8]:
            steps = 200 if reach in [1, 4] else 100
            for grid_size in [32, 64, 96, 128]:
                for variant in heat_3d_variants(include_experimental):
                    cases.append(BenchCase("heat", 3, grid_size, reach, steps, variant))
        for reach in [1, 4, 8]:
            for grid_size in [128, 256, 512, 1024]:
                for variant in heat_2d_variants(include_experimental):
                    cases.append(BenchCase("heat", 2, grid_size, reach, 500, variant))
        return cases

    if profile == "cinematic":
        # One fixed setup, designed for the animated story.
        for variant in heat_3d_variants(include_experimental):
            cases.append(BenchCase("heat", 3, 128, 4, 100, variant))
        for variant in heat_2d_variants(include_experimental):
            cases.append(BenchCase("heat", 2, 512, 4, 300, variant))
        return cases

    if profile == "large":
        for reach in [1, 4, 8]:
            for grid_size in [160, 192, 224, 256]:
                variants = ["fp32", "kahan", "fp32_25d", "kahan_25d", "kahan_25d_async"]
                if include_experimental:
                    variants.append("cfp16_kahan_3d_tiled")
                for variant in variants:
                    cases.append(BenchCase("heat", 3, grid_size, reach, 50, variant))
        return cases

    if profile == "serious":
        three_d_schedule = [
            (160, 500),
            (192, 450),
            (256, 350),
            (320, 250),
            (384, 180),
            (448, 120),
            (512, 80),
        ]
        two_d_schedule = [
            (1024, 5000),
            (2048, 3500),
            (4096, 2000),
            (8192, 800),
        ]

        for grid_size, timesteps in three_d_schedule:
            for reach in [1, 4, 8]:
                for variant in heat_3d_variants(include_experimental):
                    cases.append(BenchCase("heat", 3, grid_size, reach, timesteps, variant))
                if include_wave:
                    for variant in wave_3d_variants():
                        cases.append(BenchCase("wave", 3, grid_size, reach, timesteps, variant))

        for grid_size, timesteps in two_d_schedule:
            for reach in [1, 4, 8]:
                for variant in heat_2d_variants(include_experimental):
                    cases.append(BenchCase("heat", 2, grid_size, reach, timesteps, variant))
                if include_wave:
                    for variant in wave_2d_variants():
                        cases.append(BenchCase("wave", 2, grid_size, reach, timesteps, variant))
        return cases

    if profile == "variance":
        three_d_schedule = [
            (192, 1000),
            (256, 900),
            (384, 500),
            (512, 250),
        ]
        two_d_schedule = [
            (1024, 10000),
            (2048, 8000),
            (4096, 5000),
            (8192, 2000),
        ]
        for grid_size, timesteps in three_d_schedule:
            for reach in [1, 4, 8]:
                for variant in heat_3d_variants(include_experimental):
                    cases.append(BenchCase("heat", 3, grid_size, reach, timesteps, variant))
        for grid_size, timesteps in two_d_schedule:
            for reach in [1, 4, 8]:
                for variant in heat_2d_variants(include_experimental):
                    cases.append(BenchCase("heat", 2, grid_size, reach, timesteps, variant))
        return cases

    if profile == "big3d":
        three_d_schedule = [
            (576, 180),
            (640, 140),
            (704, 100),
            (768, 80),
        ]
        for grid_size, timesteps in three_d_schedule:
            for reach in [1, 4, 8]:
                for variant in heat_3d_variants(include_experimental):
                    cases.append(BenchCase("heat", 3, grid_size, reach, timesteps, variant))
        return cases

    if profile == "hero1024":
        for variant in ["fp16"]:
            cases.append(BenchCase("heat", 3, 1024, 1, 50, variant))
        return cases

    if profile == "zreg":
        variants = ["fp32_25d", "fp32_25d_zreg", "fp16", "fp16_25d_zreg", "fp16_25d_zreg_async", "kahan_25d", "kahan_25d_zreg", "kahan_25d_zreg_async", "kahan_25d_async"]
        for grid_size, timesteps in [(128, 100), (256, 200), (384, 150), (512, 100), (640, 80), (768, 60)]:
            for reach in [1, 4, 8]:
                for variant in variants:
                    cases.append(BenchCase("heat", 3, grid_size, reach, timesteps, variant))
        return cases

    if profile == "async_zreg":
        variants = ["fp16", "fp16_25d_zreg", "fp16_25d_zreg_async", "kahan_25d", "kahan_25d_zreg", "kahan_25d_zreg_async", "kahan_25d_async"]
        for grid_size, timesteps in [(256, 200), (512, 100), (640, 80), (768, 60)]:
            for reach in [1, 4, 8]:
                for variant in variants:
                    cases.append(BenchCase("heat", 3, grid_size, reach, timesteps, variant))
        return cases

    raise ValueError(f"unknown profile: {profile}")


def make_run_row(
    *,
    run_id: str,
    profile: str,
    case: BenchCase,
    repeat: int,
    warmup_runs: int,
    metadata: dict[str, str],
    status: str,
    exit_code: int,
    duration_wall_sec: float,
    stdout_log: Path,
    stderr_log: Path,
    result_row: dict[str, str] | None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "profile": profile,
        "project": case.project,
        "equation": case.equation,
        "git_commit": metadata["git_commit"],
        "gpu_name": metadata["gpu_name"],
        "compute_cap": metadata["compute_cap"],
        "driver_version": metadata["driver_version"],
        "cuda_arch": metadata["cuda_arch"],
        "variant_request": case.variant,
        "variant": "",
        "dim": case.dim,
        "grid_size": case.grid_size,
        "reach": case.reach,
        "timesteps": case.timesteps,
        "repeat": repeat,
        "warmup_runs": warmup_runs,
        "elapsed_ms": "",
        "megapoints_per_sec": "",
        "bandwidth_gbs": "",
        "max_abs_error": "",
        "l2_error": "",
        "memory_bytes": "",
        "status": status,
        "exit_code": exit_code,
        "duration_wall_sec": f"{duration_wall_sec:.6f}",
        "stdout_log": str(stdout_log.relative_to(ROOT)),
        "stderr_log": str(stderr_log.relative_to(ROOT)),
    }
    if result_row:
        row.update(
            {
                "variant": result_row.get("variant", ""),
                "dim": result_row.get("dim", case.dim),
                "reach": result_row.get("reach", case.reach),
                "grid_size": result_row.get("grid_size", case.grid_size),
                "timesteps": result_row.get("timesteps", case.timesteps),
                "elapsed_ms": result_row.get("elapsed_ms", ""),
                "max_abs_error": result_row.get("max_abs_error", ""),
                "l2_error": result_row.get("l2_error", ""),
                "bandwidth_gbs": result_row.get("bandwidth_gbs", ""),
                "megapoints_per_sec": derived_mpoints(case, result_row),
                "memory_bytes": result_row.get("memory_bytes", ""),
            }
        )
    return row


def run_case(
    *,
    case: BenchCase,
    repeat: int,
    profile: str,
    build_name: str,
    out_dir: Path,
    raw_csv: Path,
    metadata: dict[str, str],
    warmup_runs: int,
    timeout: int,
    dry_run: bool,
    keep_cpu_reference: bool,
    skip_cpu_reference: bool,
) -> None:
    run_id = run_id_for(case, repeat)
    run_safe = run_id.replace("/", "_")
    logs_dir = out_dir / "logs" / case.project
    tmp_dir = out_dir / "tmp"
    logs_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    stdout_log = logs_dir / f"{run_safe}.stdout.log"
    stderr_log = logs_dir / f"{run_safe}.stderr.log"
    tmp_csv = tmp_dir / f"{run_safe}.csv"
    tmp_csv.unlink(missing_ok=True)

    exe = binary_path(case.project, build_name)
    cmd = [
        str(exe),
        "-n",
        str(case.grid_size),
        "-t",
        str(case.timesteps),
        "-d",
        str(case.dim),
        "-r",
        str(case.reach),
        "-v",
        case.variant,
        "-o",
        str(tmp_csv),
    ]
    if skip_cpu_reference and case.project == "heat":
        cmd.append("--no-reference")

    if dry_run:
        print(command_text(cmd))
        return

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(project_dir(case.project)),
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        status = "ok" if proc.returncode == 0 else "failed"
        exit_code = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        exit_code = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
    duration = time.perf_counter() - start

    stdout_log.write_text(stdout, encoding="utf-8")
    stderr_log.write_text(stderr, encoding="utf-8")

    result_rows = read_result_rows(tmp_csv)
    if status == "ok" and not result_rows:
        status = "no_csv_rows"

    rows: list[dict[str, object]] = []
    if result_rows:
        non_cpu_rows = [row for row in result_rows if "cpu" not in row.get("variant", "")]
        cpu_rows = [row for row in result_rows if "cpu" in row.get("variant", "")]
        selected_rows = non_cpu_rows or result_rows
        for result_row in selected_rows:
            rows.append(
                make_run_row(
                    run_id=run_id,
                    profile=profile,
                    case=case,
                    repeat=repeat,
                    warmup_runs=warmup_runs,
                    metadata=metadata,
                    status=status,
                    exit_code=exit_code,
                    duration_wall_sec=duration,
                    stdout_log=stdout_log,
                    stderr_log=stderr_log,
                    result_row=result_row,
                )
            )
        if keep_cpu_reference and repeat == 1 and cpu_rows:
            cpu_run_id = f"{case.project}_{case.dim}d_n{case.grid_size}_r{case.reach}_t{case.timesteps}_cpu_ref"
            rows.append(
                make_run_row(
                    run_id=cpu_run_id,
                    profile=profile,
                    case=case,
                    repeat=1,
                    warmup_runs=warmup_runs,
                    metadata=metadata,
                    status=status,
                    exit_code=exit_code,
                    duration_wall_sec=duration,
                    stdout_log=stdout_log,
                    stderr_log=stderr_log,
                    result_row=cpu_rows[0],
                )
            )
    else:
        rows.append(
            make_run_row(
                run_id=run_id,
                profile=profile,
                case=case,
                repeat=repeat,
                warmup_runs=warmup_runs,
                metadata=metadata,
                status=status,
                exit_code=exit_code,
                duration_wall_sec=duration,
                stdout_log=stdout_log,
                stderr_log=stderr_log,
                result_row=None,
            )
        )
    append_csv(raw_csv, RAW_COLUMNS, rows)


def summarize(raw_csv: Path, summary_csv: Path) -> list[dict[str, object]]:
    if not raw_csv.exists():
        return []
    with raw_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    latest_rows: dict[str, dict[str, str]] = {}
    for row in rows:
        run_id = row.get("run_id", "")
        if run_id:
            latest_rows[run_id] = row
    rows = list(latest_rows.values())

    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row["profile"],
            row["project"],
            row["equation"],
            row["variant"],
            row["dim"],
            row["grid_size"],
            row["reach"],
            row["timesteps"],
        )
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, object]] = []
    for key, items in sorted(groups.items()):
        ok_items = [row for row in items if row["status"] == "ok" and row["elapsed_ms"]]
        failed = len(items) - len(ok_items)

        elapsed = [maybe_float(row["elapsed_ms"]) for row in ok_items]
        mpoints = [maybe_float(row["megapoints_per_sec"]) for row in ok_items]
        max_errors = [maybe_float(row["max_abs_error"]) for row in ok_items]
        l2_errors = [maybe_float(row["l2_error"]) for row in ok_items]

        def mean(values: list[float]) -> float:
            return statistics.mean(values) if values else 0.0

        def median(values: list[float]) -> float:
            return statistics.median(values) if values else 0.0

        def std(values: list[float]) -> float:
            return statistics.stdev(values) if len(values) > 1 else 0.0

        row = {
            "profile": key[0],
            "project": key[1],
            "equation": key[2],
            "variant": key[3],
            "dim": key[4],
            "grid_size": key[5],
            "reach": key[6],
            "timesteps": key[7],
            "runs_ok": len(ok_items),
            "runs_failed": failed,
            "elapsed_ms_mean": mean(elapsed),
            "elapsed_ms_median": median(elapsed),
            "elapsed_ms_std": std(elapsed),
            "elapsed_ms_min": min(elapsed) if elapsed else 0.0,
            "elapsed_ms_max": max(elapsed) if elapsed else 0.0,
            "mpoints_mean": mean(mpoints),
            "mpoints_median": median(mpoints),
            "mpoints_std": std(mpoints),
            "mpoints_min": min(mpoints) if mpoints else 0.0,
            "mpoints_max": max(mpoints) if mpoints else 0.0,
            "max_abs_error_mean": mean(max_errors),
            "l2_error_mean": mean(l2_errors),
        }
        summary_rows.append(row)

    summary_csv.unlink(missing_ok=True)
    append_csv(summary_csv, SUMMARY_COLUMNS, summary_rows)
    return summary_rows


def write_cinematic_json(summary_rows: list[dict[str, object]], path: Path) -> None:
    stage_order = [
        ("cpu_fp64_3d", "CPU fp64 reference", "Correct but slow reference"),
        ("cuda_fp32_3d", "fp32 baseline", "Standard GPU baseline"),
        ("cuda_fp16_naive_3d", "fp16 naive", "Fast storage, bad error growth"),
        ("cuda_fp16_kahan_3d", "fp16 + Kahan", "Accuracy recovered"),
        ("cuda_fp32_3d_25d", "fp32 + 2.5D", "Shared-memory XY tiling"),
        ("cuda_fp16_kahan_3d_25d", "Kahan + 2.5D", "Accuracy plus memory tiling"),
        ("cuda_fp16_kahan_3d_25d_async", "Kahan + 2.5D async", "Async-copy experiment"),
    ]
    rows = [row for row in summary_rows if str(row.get("profile")) == "cinematic"]
    if not rows:
        rows = summary_rows

    stages = []
    for index, (variant, name, description) in enumerate(stage_order):
        candidates = [row for row in rows if row.get("variant") == variant]
        if not candidates:
            continue
        # Prefer the fixed 3D cinematic setup if present.
        candidates.sort(key=lambda row: (str(row.get("dim")) != "3", str(row.get("grid_size")) != "128"))
        row = candidates[0]
        stages.append(
            {
                "stage": index,
                "name": name,
                "variant": variant,
                "description": description,
                "dim": int(row["dim"]),
                "grid_size": int(row["grid_size"]),
                "reach": int(row["reach"]),
                "timesteps": int(row["timesteps"]),
                "mpoints_per_sec": maybe_float(row["mpoints_median"]),
                "elapsed_ms": maybe_float(row["elapsed_ms_median"]),
                "max_abs_error": maybe_float(row["max_abs_error_mean"]),
                "l2_error": maybe_float(row["l2_error_mean"]),
            }
        )

    write_json(path, {"created_at": datetime.now().isoformat(timespec="seconds"), "stages": stages})


def main() -> int:
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    parser = argparse.ArgumentParser(description="Run final CUDA benchmark matrix with robust logging.")
    parser.add_argument("--profile", choices=["smoke", "calibration", "validated", "cinematic", "large", "serious", "variance", "big3d", "hero1024", "zreg", "async_zreg"], default="smoke")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--build-name", default="build-final-rtx4070")
    parser.add_argument("--cuda-arch", default="89")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--max-hours", type=float, default=None, help="Stop cleanly before starting a new case after this many wall-clock hours.")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--include-wave", action="store_true")
    parser.add_argument("--include-experimental", action="store_true", help="Include CFP16 heat variants where exposed.")
    parser.add_argument("--keep-cpu-reference", action="store_true", help="Keep one CPU reference row per project/dim/grid/reach/timestep config in raw and summary outputs.")
    parser.add_argument("--skip-cpu-reference", action="store_true", help="Pass --no-reference to heat_stencil for GPU-only timing runs.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle case order with a deterministic seed to reduce order bias.")
    parser.add_argument("--seed", type=int, default=20260507, help="Seed used with --shuffle.")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Skip run IDs whose latest raw row is already ok.")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.out_dir is None:
        out_dir = DEFAULT_OUT / f"{timestamp}_{args.profile}_rtx4070"
    elif args.out_dir.is_absolute():
        out_dir = args.out_dir
    else:
        out_dir = ROOT / args.out_dir
    out_dir = out_dir.resolve()
    raw_csv = out_dir / "raw_runs.csv"
    summary_csv = out_dir / "summary_stats.csv"
    cinematic_json = out_dir / "cinematic_timeline.json"

    repeats = args.repeats
    if repeats is None:
        repeats = 1 if args.profile in ["smoke", "calibration"] else 5

    cases = make_cases(args.profile, args.include_wave, args.include_experimental)
    if args.shuffle:
        random.Random(args.seed).shuffle(cases)
    projects = {case.project for case in cases}
    metadata = {
        "git_commit": get_git_commit(),
        "cuda_arch": args.cuda_arch,
        **get_gpu_info(),
        "python": sys.version,
        "platform": platform.platform(),
        "profile": args.profile,
        "repeats": str(repeats),
        "warmup_runs": str(args.warmup_runs),
        "keep_cpu_reference": str(args.keep_cpu_reference),
        "skip_cpu_reference": str(args.skip_cpu_reference),
        "shuffle": str(args.shuffle),
        "seed": str(args.seed),
        "cases": [asdict(case) for case in cases],
    }
    write_json(out_dir / "metadata.json", metadata)

    print(f"output: {out_dir}")
    print(f"profile: {args.profile}, cases: {len(cases)}, repeats: {repeats}")
    print(f"gpu: {metadata['gpu_name']} cc {metadata['compute_cap']}, driver {metadata['driver_version']}")
    print(f"resume: {'on' if args.resume else 'off'}")
    print(f"shuffle: {'on' if args.shuffle else 'off'} seed={args.seed}")
    print(f"keep cpu reference: {'on' if args.keep_cpu_reference else 'off'}")
    print(f"skip cpu reference: {'on' if args.skip_cpu_reference else 'off'}")

    if not args.no_build and not args.dry_run:
        build_projects(projects, args.build_name, args.cuda_arch)

    total = len(cases) * repeats
    completed = completed_run_ids(raw_csv) if args.resume and not args.dry_run else set()
    if completed:
        print(f"resume: skipping {len(completed)} already-ok runs from {raw_csv}")
    iterator = ((case, repeat) for repeat in range(1, repeats + 1) for case in cases)
    progress = tqdm(iterator, total=total, desc=args.profile, unit="run") if tqdm else iterator

    started = time.perf_counter()
    skipped = 0
    for case, repeat in progress:
        if STOP_REQUESTED:
            print("stop requested; stopping before next case", flush=True)
            break

        if args.max_hours is not None and (time.perf_counter() - started) >= args.max_hours * 3600.0:
            print(f"max-hours reached ({args.max_hours:g}h); stopping before next case", flush=True)
            break

        run_id = run_id_for(case, repeat)
        if run_id in completed:
            skipped += 1
            if tqdm:
                progress.set_postfix_str(f"skip {case.project} {case.dim}D N={case.grid_size} R={case.reach} {case.variant}")
            else:
                print(f"[skip] {run_id}")
            continue

        if tqdm:
            progress.set_postfix_str(f"{case.project} {case.dim}D N={case.grid_size} R={case.reach} {case.variant}")
        else:
            done_guess = time.perf_counter() - started
            print(f"[{repeat}/{repeats}] {case.project} {case.dim}D N={case.grid_size} R={case.reach} {case.variant} ({done_guess:.1f}s elapsed)")

        # warmups are intentionally not logged as benchmark rows.
        for warmup in range(args.warmup_runs):
            run_case(
                case=case,
                repeat=-(warmup + 1),
                profile=args.profile,
                build_name=args.build_name,
                out_dir=out_dir,
                raw_csv=out_dir / "warmup_runs.csv",
                metadata=metadata,
                warmup_runs=args.warmup_runs,
                timeout=args.timeout,
                dry_run=args.dry_run,
                keep_cpu_reference=False,
                skip_cpu_reference=args.skip_cpu_reference,
            )

        run_case(
            case=case,
            repeat=repeat,
            profile=args.profile,
            build_name=args.build_name,
            out_dir=out_dir,
            raw_csv=raw_csv,
            metadata=metadata,
            warmup_runs=args.warmup_runs,
            timeout=args.timeout,
            dry_run=args.dry_run,
            keep_cpu_reference=args.keep_cpu_reference,
            skip_cpu_reference=args.skip_cpu_reference,
        )

    if args.dry_run:
        return 0

    summary_rows = summarize(raw_csv, summary_csv)
    write_cinematic_json(summary_rows, cinematic_json)
    if skipped:
        print(f"skipped already-ok runs: {skipped}")
    print(f"raw rows: {raw_csv}")
    print(f"summary:  {summary_csv}")
    print(f"cinema:   {cinematic_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())