#!/usr/bin/env python3
import csv
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSIGNMENT_BIN = ROOT / "assignment2"
OUTPUT_CSV = ROOT / "benchmark_large_results.csv"
DATA_DIR = ROOT / "benchmark_data_large"

LARGE_CASES = [
    {"name": "large_32768_k32_t20", "n": 32768, "k": 32, "t": 20, "seed": 707},
    {"name": "large_50000_k64_t30", "n": 50000, "k": 64, "t": 30, "seed": 808},
    {"name": "large_75000_k96_t40", "n": 75000, "k": 96, "t": 40, "seed": 909},
    {"name": "large_100000_k128_t50", "n": 100000, "k": 128, "t": 50, "seed": 1001},
]
WARMUP_RUNS = 1
TIMED_RUNS = 2
GPU_HOST_THREADS = min(16, os.cpu_count() or 1)


def run_command(cmd, env=None, capture_output=False):
    return subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def generate_dataset(path, n, k, t, seed):
    rng = random.Random(seed)
    used = set()
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{n}\n{k}\n{t}\n")
        while len(used) < n:
            point = (
                rng.randint(-10000, 10000),
                rng.randint(-10000, 10000),
                rng.randint(-10000, 10000),
            )
            if point in used:
                continue
            used.add(point)
            intensity = rng.randint(0, 255)
            handle.write(f"{point[0]} {point[1]} {point[2]} {intensity}\n")


def build_binary():
    run_command(["make", "assignment2"])


def benchmark_case(case):
    input_path = DATA_DIR / f"{case['name']}.txt"
    generate_dataset(input_path, case["n"], case["k"], case["t"], case["seed"])

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(GPU_HOST_THREADS)
    result = run_command(
        [
            str(ASSIGNMENT_BIN),
            "--benchmark-gpu-only",
            "--warmups",
            str(WARMUP_RUNS),
            "--repeats",
            str(TIMED_RUNS),
            str(input_path),
        ],
        env=env,
        capture_output=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"No benchmark JSON produced for {case['name']}")
    report = json.loads(lines[-1])
    report["case"] = case["name"]
    report["exact_over_approx_end_to_end"] = (
        report["exact_gpu_end_to_end_ms"] / report["approx_gpu_end_to_end_ms"]
        if report["approx_gpu_end_to_end_ms"] > 0.0 else 0.0
    )
    report["exact_over_approx_kernel"] = (
        report["exact_gpu_kernel_ms"] / report["approx_gpu_kernel_ms"]
        if report["approx_gpu_kernel_ms"] > 0.0 else 0.0
    )
    return report


def format_float(value):
    return f"{value:.6f}"


def main():
    DATA_DIR.mkdir(exist_ok=True)
    build_binary()

    rows = [benchmark_case(case) for case in LARGE_CASES]

    fieldnames = [
        "case", "n", "k", "T", "gpu_host_threads", "warmups", "repeats",
        "cold_start_ms", "shared_input_upload_ms", "shared_knn_preprocess_ms", "shared_knn_upload_ms",
        "exact_gpu_end_to_end_ms", "exact_gpu_kernel_ms",
        "approx_gpu_preprocess_ms", "approx_gpu_upload_ms", "approx_gpu_end_to_end_ms", "approx_gpu_kernel_ms",
        "approx_fallback_count", "exact_over_approx_end_to_end", "exact_over_approx_kernel",
        "kmeans_gpu_end_to_end_ms", "kmeans_gpu_kernel_ms",
    ]

    normalized_rows = []
    for row in rows:
        normalized_rows.append({
            "case": row["case"],
            "n": row["n"],
            "k": row["k"],
            "T": row["T"],
            "gpu_host_threads": row["gpu_host_threads"],
            "warmups": row["warmups"],
            "repeats": row["repeats"],
            "cold_start_ms": row["cold_start_ms"],
            "shared_input_upload_ms": row["shared_input_upload_ms"],
            "shared_knn_preprocess_ms": row["shared_knn_preprocess_ms"],
            "shared_knn_upload_ms": row["shared_knn_upload_ms"],
            "exact_gpu_end_to_end_ms": row["exact_gpu_end_to_end_ms"],
            "exact_gpu_kernel_ms": row["exact_gpu_kernel_ms"],
            "approx_gpu_preprocess_ms": row["approx_gpu_preprocess_ms"],
            "approx_gpu_upload_ms": row["approx_gpu_upload_ms"],
            "approx_gpu_end_to_end_ms": row["approx_gpu_end_to_end_ms"],
            "approx_gpu_kernel_ms": row["approx_gpu_kernel_ms"],
            "approx_fallback_count": row["approx_fallback_count"],
            "exact_over_approx_end_to_end": row["exact_over_approx_end_to_end"],
            "exact_over_approx_kernel": row["exact_over_approx_kernel"],
            "kmeans_gpu_end_to_end_ms": row["kmeans_gpu_end_to_end_ms"],
            "kmeans_gpu_kernel_ms": row["kmeans_gpu_kernel_ms"],
        })

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in normalized_rows:
            writer.writerow(row)

    print(f"Wrote {OUTPUT_CSV}")
    print(f"GPU host threads: {GPU_HOST_THREADS}")
    for row in normalized_rows:
        print(
            f"{row['case']}: "
            f"shared_knn_build={format_float(row['shared_knn_preprocess_ms'])}ms, "
            f"shared_knn_upload={format_float(row['shared_knn_upload_ms'])}ms, "
            f"exact_ms={format_float(row['exact_gpu_end_to_end_ms'])}, "
            f"approx_ms={format_float(row['approx_gpu_end_to_end_ms'])}, "
            f"exact_over_approx={format_float(row['exact_over_approx_end_to_end'])}x, "
            f"fallbacks={row['approx_fallback_count']}, "
            f"kmeans_ms={format_float(row['kmeans_gpu_end_to_end_ms'])}"
        )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(error.stderr or error.stdout or str(error), file=sys.stderr)
        sys.exit(error.returncode)
