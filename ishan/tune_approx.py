#!/usr/bin/env python3
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSIGNMENT_BIN = ROOT / "assignment2"
SUBMISSION_BIN = ROOT / "a2"
DATA_DIR = ROOT / "tuning_data"
REF_DIR = ROOT / "tuning_refs"
OUT_DIR = ROOT / "tuning_outputs"
DETAIL_CSV = ROOT / "approx_tuning_results.csv"
SUMMARY_CSV = ROOT / "approx_tuning_summary.csv"
FINAL_CSV = ROOT / "approx_tuning_best_results.csv"
GPU_HOST_THREADS = min(16, os.cpu_count() or 1)
WARMUPS = 1
REPEATS = 2

TRAINING_CASES = [
    {"name": "tune_4096_k16_t15", "n": 4096, "k": 16, "t": 15, "seed": 1201},
    {"name": "tune_8192_k32_t20", "n": 8192, "k": 32, "t": 20, "seed": 1202},
    {"name": "tune_16384_k64_t30", "n": 16384, "k": 64, "t": 30, "seed": 1203},
    {"name": "tune_32768_k32_t20", "n": 32768, "k": 32, "t": 20, "seed": 1204},
    {"name": "tune_50000_k64_t30", "n": 50000, "k": 64, "t": 30, "seed": 1205},
]

FINAL_CASES = TRAINING_CASES + [
    {"name": "tune_75000_k96_t40", "n": 75000, "k": 96, "t": 40, "seed": 1206},
    {"name": "tune_100000_k128_t50", "n": 100000, "k": 128, "t": 50, "seed": 1207},
]

CONFIGS = [
    {"name": "baseline_2p00", "second_shell": 1, "third_shell": 0, "accept_num": 2, "accept_den": 1, "third_trigger_num": 1, "third_trigger_den": 1},
    {"name": "accept_1p50", "second_shell": 1, "third_shell": 0, "accept_num": 3, "accept_den": 2, "third_trigger_num": 1, "third_trigger_den": 1},
    {"name": "accept_1p25", "second_shell": 1, "third_shell": 0, "accept_num": 5, "accept_den": 4, "third_trigger_num": 1, "third_trigger_den": 1},
    {"name": "accept_1p00", "second_shell": 1, "third_shell": 0, "accept_num": 1, "accept_den": 1, "third_trigger_num": 1, "third_trigger_den": 1},
    {"name": "own_cell_only", "second_shell": 0, "third_shell": 0, "accept_num": 1, "accept_den": 1, "third_trigger_num": 1, "third_trigger_den": 1},
    {"name": "accept_1p50_shell2", "second_shell": 1, "third_shell": 1, "accept_num": 3, "accept_den": 2, "third_trigger_num": 1, "third_trigger_den": 1},
    {"name": "accept_1p25_shell2", "second_shell": 1, "third_shell": 1, "accept_num": 5, "accept_den": 4, "third_trigger_num": 1, "third_trigger_den": 1},
    {"name": "accept_1p00_shell2", "second_shell": 1, "third_shell": 1, "accept_num": 1, "accept_den": 1, "third_trigger_num": 1, "third_trigger_den": 1},
    {"name": "accept_1p25_shell2_dense", "second_shell": 1, "third_shell": 1, "accept_num": 5, "accept_den": 4, "third_trigger_num": 5, "third_trigger_den": 4},
    {"name": "accept_1p00_shell2_dense", "second_shell": 1, "third_shell": 1, "accept_num": 1, "accept_den": 1, "third_trigger_num": 5, "third_trigger_den": 4},
]


def run_command(cmd, *, cwd=ROOT, env=None, capture_output=False):
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        check=True,
        capture_output=capture_output,
    )


def generate_dataset(path: Path, n: int, k: int, t: int, seed: int) -> None:
    if path.exists():
        return
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


def ensure_datasets(cases):
    DATA_DIR.mkdir(exist_ok=True)
    for case in cases:
        generate_dataset(DATA_DIR / f"{case['name']}.txt", case['n'], case['k'], case['t'], case['seed'])


def flags_for_config(config):
    return (
        "-O3 -Xcompiler -fopenmp -std=c++14 "
        f"-DA2_APPROX_USE_SECOND_SHELL={config['second_shell']} "
        f"-DA2_APPROX_USE_THIRD_SHELL={config['third_shell']} "
        f"-DA2_APPROX_OWN_CELL_ACCEPT_NUM={config['accept_num']} "
        f"-DA2_APPROX_OWN_CELL_ACCEPT_DEN={config['accept_den']} "
        f"-DA2_APPROX_THIRD_SHELL_TRIGGER_NUM={config['third_trigger_num']} "
        f"-DA2_APPROX_THIRD_SHELL_TRIGGER_DEN={config['third_trigger_den']}"
    )


def build_config(config):
    args = ["make", "-B", "assignment2", f"ASSIGNMENT_NVCCFLAGS={flags_for_config(config)}"]
    run_command(args)


def parse_benchmark_output(stdout: str):
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("no benchmark JSON found")
    return json.loads(lines[-1])


def benchmark_case(input_path: Path):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(GPU_HOST_THREADS)
    result = run_command(
        [
            str(ASSIGNMENT_BIN),
            "--benchmark-gpu-only",
            "--warmups",
            str(WARMUPS),
            "--repeats",
            str(REPEATS),
            str(input_path),
        ],
        env=env,
        capture_output=True,
    )
    return parse_benchmark_output(result.stdout)


def parse_output_intensities(path: Path):
    values = []
    with path.open("r", encoding="utf-8", newline=None) as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            values.append(int(parts[3]))
    return values


def compute_mae(reference_path: Path, candidate_path: Path):
    reference = parse_output_intensities(reference_path)
    candidate = parse_output_intensities(candidate_path)
    if len(reference) != len(candidate):
        raise RuntimeError(f"length mismatch: {reference_path} vs {candidate_path}")
    total = 0.0
    max_abs = 0
    mismatches = 0
    for ref, cand in zip(reference, candidate):
        diff = abs(ref - cand)
        total += diff
        if diff > max_abs:
            max_abs = diff
        if diff != 0:
            mismatches += 1
    mae = total / len(reference) if reference else 0.0
    return mae, max_abs, mismatches


def ensure_exact_reference(input_path: Path, case_name: str):
    REF_DIR.mkdir(exist_ok=True)
    case_dir = REF_DIR / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    reference_path = case_dir / "knn.txt"
    if reference_path.exists():
        return reference_path
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(GPU_HOST_THREADS)
    run_command([str(SUBMISSION_BIN), str(input_path), "knn"], cwd=case_dir, env=env)
    return reference_path


def run_approx_output(input_path: Path, case_name: str, config_name: str):
    case_dir = OUT_DIR / config_name / case_name
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(GPU_HOST_THREADS)
    run_command([str(SUBMISSION_BIN), str(input_path), "approx_knn"], cwd=case_dir, env=env)
    return case_dir / "approx_knn.txt"


def geometric_mean(values):
    logs = [math.log(v) for v in values if v > 0.0]
    if not logs:
        return 0.0
    return math.exp(sum(logs) / len(logs))


def summarize_config(config, rows):
    speedups = [row["speedup_over_exact"] for row in rows]
    maes = [row["approx_mae"] for row in rows]
    fallbacks = [row["approx_fallback_count"] for row in rows]
    return {
        "config": config["name"],
        "second_shell": config["second_shell"],
        "third_shell": config["third_shell"],
        "accept_num": config["accept_num"],
        "accept_den": config["accept_den"],
        "third_trigger_num": config["third_trigger_num"],
        "third_trigger_den": config["third_trigger_den"],
        "cases": len(rows),
        "max_mae": max(maes),
        "avg_mae": sum(maes) / len(maes),
        "geomean_speedup_over_exact": geometric_mean(speedups),
        "min_speedup_over_exact": min(speedups),
        "max_speedup_over_exact": max(speedups),
        "avg_fallback_count": sum(fallbacks) / len(fallbacks),
        "max_fallback_count": max(fallbacks),
        "valid_under_mae_1": int(max(maes) < 1.0),
    }


def evaluate_configs(cases, configs):
    OUT_DIR.mkdir(exist_ok=True)
    detail_rows = []
    summary_rows = []
    for config in configs:
        print(f"Building {config['name']} ...", flush=True)
        build_config(config)
        config_rows = []
        for case in cases:
            input_path = DATA_DIR / f"{case['name']}.txt"
            reference_path = ensure_exact_reference(input_path, case['name'])
            report = benchmark_case(input_path)
            approx_output = run_approx_output(input_path, case['name'], config['name'])
            mae, max_abs, mismatches = compute_mae(reference_path, approx_output)
            exact_ms = report["exact_gpu_end_to_end_ms"]
            approx_ms = report["approx_gpu_end_to_end_ms"]
            speedup = exact_ms / approx_ms if approx_ms > 0.0 else 0.0
            row = {
                "config": config['name'],
                "case": case['name'],
                "n": case['n'],
                "k": case['k'],
                "T": case['t'],
                "second_shell": config['second_shell'],
                "third_shell": config['third_shell'],
                "accept_num": config['accept_num'],
                "accept_den": config['accept_den'],
                "third_trigger_num": config['third_trigger_num'],
                "third_trigger_den": config['third_trigger_den'],
                "exact_ms": exact_ms,
                "approx_ms": approx_ms,
                "speedup_over_exact": speedup,
                "approx_kernel_ms": report['approx_gpu_kernel_ms'],
                "approx_fallback_count": report['approx_fallback_count'],
                "approx_mae": mae,
                "approx_max_abs_error": max_abs,
                "approx_mismatch_count": mismatches,
            }
            detail_rows.append(row)
            config_rows.append(row)
            print(
                f"  {case['name']}: speedup={speedup:.3f}x, mae={mae:.6f}, fallbacks={report['approx_fallback_count']}",
                flush=True,
            )
        summary = summarize_config(config, config_rows)
        summary_rows.append(summary)
        print(
            f"Summary {config['name']}: geomean_speedup={summary['geomean_speedup_over_exact']:.3f}x, max_mae={summary['max_mae']:.6f}",
            flush=True,
        )
    return detail_rows, summary_rows


def evaluate_best_config(best_config, cases):
    build_config(best_config)
    rows = []
    for case in cases:
        input_path = DATA_DIR / f"{case['name']}.txt"
        reference_path = ensure_exact_reference(input_path, case['name'])
        report = benchmark_case(input_path)
        approx_output = run_approx_output(input_path, case['name'], best_config['name'] + "_final")
        mae, max_abs, mismatches = compute_mae(reference_path, approx_output)
        exact_ms = report['exact_gpu_end_to_end_ms']
        approx_ms = report['approx_gpu_end_to_end_ms']
        rows.append({
            'config': best_config['name'],
            'case': case['name'],
            'n': case['n'],
            'k': case['k'],
            'T': case['t'],
            'exact_ms': exact_ms,
            'approx_ms': approx_ms,
            'speedup_over_exact': exact_ms / approx_ms if approx_ms > 0.0 else 0.0,
            'approx_mae': mae,
            'approx_max_abs_error': max_abs,
            'approx_mismatch_count': mismatches,
            'approx_fallback_count': report['approx_fallback_count'],
        })
    return rows


def write_csv(path: Path, rows, fieldnames):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    ensure_datasets(FINAL_CASES)
    build_config(CONFIGS[0])
    for case in FINAL_CASES:
        input_path = DATA_DIR / f"{case['name']}.txt"
        ensure_exact_reference(input_path, case['name'])

    detail_rows, summary_rows = evaluate_configs(TRAINING_CASES, CONFIGS)
    write_csv(
        DETAIL_CSV,
        detail_rows,
        [
            'config', 'case', 'n', 'k', 'T', 'second_shell', 'third_shell',
            'accept_num', 'accept_den', 'third_trigger_num', 'third_trigger_den',
            'exact_ms', 'approx_ms', 'speedup_over_exact', 'approx_kernel_ms',
            'approx_fallback_count', 'approx_mae', 'approx_max_abs_error', 'approx_mismatch_count',
        ],
    )
    summary_rows.sort(key=lambda row: (-row['valid_under_mae_1'], -row['geomean_speedup_over_exact'], row['max_mae']))
    write_csv(
        SUMMARY_CSV,
        summary_rows,
        [
            'config', 'second_shell', 'third_shell', 'accept_num', 'accept_den',
            'third_trigger_num', 'third_trigger_den', 'cases', 'max_mae', 'avg_mae',
            'geomean_speedup_over_exact', 'min_speedup_over_exact', 'max_speedup_over_exact',
            'avg_fallback_count', 'max_fallback_count', 'valid_under_mae_1',
        ],
    )

    valid = [row for row in summary_rows if row['valid_under_mae_1'] == 1]
    if not valid:
        raise RuntimeError('No configuration satisfied max MAE < 1.0')
    best_summary = valid[0]
    best_config = next(config for config in CONFIGS if config['name'] == best_summary['config'])
    print(f"Best config: {best_config['name']}", flush=True)

    final_rows = evaluate_best_config(best_config, FINAL_CASES)
    write_csv(
        FINAL_CSV,
        final_rows,
        [
            'config', 'case', 'n', 'k', 'T', 'exact_ms', 'approx_ms',
            'speedup_over_exact', 'approx_mae', 'approx_max_abs_error',
            'approx_mismatch_count', 'approx_fallback_count',
        ],
    )
    print(f"Wrote {DETAIL_CSV}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {FINAL_CSV}")


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(error.stderr or error.stdout or str(error), file=sys.stderr)
        sys.exit(error.returncode)
