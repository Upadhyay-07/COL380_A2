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
OUTPUT_CSV = ROOT / "benchmark_results.csv"
DATA_DIR = ROOT / "benchmark_data"

CASES = [
    {"name": "case_2048_k8_t10", "n": 2048, "k": 8, "t": 10, "seed": 101},
    {"name": "case_4096_k16_t15", "n": 4096, "k": 16, "t": 15, "seed": 202},
    {"name": "case_8192_k32_t20", "n": 8192, "k": 32, "t": 20, "seed": 303},
    {"name": "case_8192_k128_t50", "n": 8192, "k": 128, "t": 50, "seed": 404},
    {"name": "case_16384_k64_t30", "n": 16384, "k": 64, "t": 30, "seed": 505},
    {"name": "case_24576_k32_t20", "n": 24576, "k": 32, "t": 20, "seed": 606},
]
WARMUP_RUNS = 1
TIMED_RUNS = 3
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
            "--benchmark",
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

    if report["exact_mismatch_count"] != 0:
        raise RuntimeError(f"Exact KNN mismatch on {case['name']}")
    if report["kmeans_mismatch_count"] != 0:
        raise RuntimeError(f"KMeans mismatch on {case['name']}")
    if report["approx_mae_vs_exact"] > 3.0:
        raise RuntimeError(
            f"Approximate KNN MAE exceeded threshold on {case['name']}: {report['approx_mae_vs_exact']}"
        )

    return report


def format_float(value):
    return f"{value:.6f}"


def main():
    DATA_DIR.mkdir(exist_ok=True)
    build_binary()

    rows = [benchmark_case(case) for case in CASES]

    fieldnames = [
        "case", "n", "k", "T", "seq_threads", "gpu_host_threads", "warmups", "repeats",
        "cold_start_ms", "shared_input_upload_ms", "shared_knn_preprocess_ms", "shared_knn_upload_ms",
        "exact_cpu_ms", "exact_gpu_end_to_end_ms", "exact_gpu_kernel_ms",
        "exact_speedup_end_to_end", "exact_speedup_compute_only",
        "exact_mae", "exact_max_abs_error", "exact_mismatch_count",
        "approx_gpu_preprocess_ms", "approx_gpu_upload_ms", "approx_gpu_end_to_end_ms",
        "approx_gpu_kernel_ms", "approx_speedup_end_to_end_vs_exact_cpu",
        "approx_speedup_compute_only_vs_exact_cpu", "approx_mae_vs_exact",
        "approx_max_abs_error", "approx_mismatch_count", "approx_fallback_count",
        "kmeans_cpu_ms", "kmeans_gpu_end_to_end_ms", "kmeans_gpu_kernel_ms",
        "kmeans_speedup_end_to_end", "kmeans_speedup_compute_only",
        "kmeans_mae", "kmeans_max_abs_error", "kmeans_mismatch_count",
    ]

    normalized_rows = []
    for row in rows:
        normalized_rows.append({
            "case": row["case"],
            "n": row["n"],
            "k": row["k"],
            "T": row["T"],
            "seq_threads": row["seq_threads"],
            "gpu_host_threads": row["gpu_host_threads"],
            "warmups": row["warmups"],
            "repeats": row["repeats"],
            "cold_start_ms": row["cold_start_ms"],
            "shared_input_upload_ms": row["shared_input_upload_ms"],
            "shared_knn_preprocess_ms": row["shared_knn_preprocess_ms"],
            "shared_knn_upload_ms": row["shared_knn_upload_ms"],
            "exact_cpu_ms": row["exact_cpu_ms"],
            "exact_gpu_end_to_end_ms": row["exact_gpu_end_to_end_ms"],
            "exact_gpu_kernel_ms": row["exact_gpu_kernel_ms"],
            "exact_speedup_end_to_end": row["exact_speedup_end_to_end"],
            "exact_speedup_compute_only": row["exact_speedup_compute_only"],
            "exact_mae": row["exact_mae"],
            "exact_max_abs_error": row["exact_max_abs_error"],
            "exact_mismatch_count": row["exact_mismatch_count"],
            "approx_gpu_preprocess_ms": row["approx_gpu_preprocess_ms"],
            "approx_gpu_upload_ms": row["approx_gpu_upload_ms"],
            "approx_gpu_end_to_end_ms": row["approx_gpu_end_to_end_ms"],
            "approx_gpu_kernel_ms": row["approx_gpu_kernel_ms"],
            "approx_speedup_end_to_end_vs_exact_cpu": row["approx_speedup_end_to_end_vs_exact_cpu"],
            "approx_speedup_compute_only_vs_exact_cpu": row["approx_speedup_compute_only_vs_exact_cpu"],
            "approx_mae_vs_exact": row["approx_mae_vs_exact"],
            "approx_max_abs_error": row["approx_max_abs_error"],
            "approx_mismatch_count": row["approx_mismatch_count"],
            "approx_fallback_count": row["approx_fallback_count"],
            "kmeans_cpu_ms": row["kmeans_cpu_ms"],
            "kmeans_gpu_end_to_end_ms": row["kmeans_gpu_end_to_end_ms"],
            "kmeans_gpu_kernel_ms": row["kmeans_gpu_kernel_ms"],
            "kmeans_speedup_end_to_end": row["kmeans_speedup_end_to_end"],
            "kmeans_speedup_compute_only": row["kmeans_speedup_compute_only"],
            "kmeans_mae": row["kmeans_mae"],
            "kmeans_max_abs_error": row["kmeans_max_abs_error"],
            "kmeans_mismatch_count": row["kmeans_mismatch_count"],
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
            f"exact_end={format_float(row['exact_speedup_end_to_end'])}x, "
            f"exact_kernel={format_float(row['exact_speedup_compute_only'])}x, "
            f"approx_end={format_float(row['approx_speedup_end_to_end_vs_exact_cpu'])}x, "
            f"approx_kernel={format_float(row['approx_speedup_compute_only_vs_exact_cpu'])}x, "
            f"kmeans_end={format_float(row['kmeans_speedup_end_to_end'])}x, "
            f"kmeans_kernel={format_float(row['kmeans_speedup_compute_only'])}x, "
            f"approx_mae={format_float(row['approx_mae_vs_exact'])}, "
            f"fallbacks={row['approx_fallback_count']}"
        )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(error.stderr or error.stdout or str(error), file=sys.stderr)
        sys.exit(error.returncode)
