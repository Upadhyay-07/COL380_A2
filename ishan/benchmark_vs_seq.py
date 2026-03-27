#!/usr/bin/env python3
import csv
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEQ_BIN = Path('/tmp/seq_mode')
A2_BIN = ROOT / 'a2'
MAE_SCRIPT = ROOT / 'mae_loss.py'
OUTPUT_CSV = ROOT / 'seq_baseline_benchmark.csv'
TMP_ROOT = ROOT / 'seq_benchmark_tmp'
GPU_HOST_THREADS = min(16, os.cpu_count() or 1)

CASES = [
    {'name': 'tune_4096_k16_t15', 'path': ROOT / 'tuning_data' / 'tune_4096_k16_t15.txt', 'n': 4096, 'k': 16, 'T': 15},
    {'name': 'tune_8192_k32_t20', 'path': ROOT / 'tuning_data' / 'tune_8192_k32_t20.txt', 'n': 8192, 'k': 32, 'T': 20},
    {'name': 'tune_16384_k64_t30', 'path': ROOT / 'tuning_data' / 'tune_16384_k64_t30.txt', 'n': 16384, 'k': 64, 'T': 30},
    {'name': 'tune_32768_k32_t20', 'path': ROOT / 'tuning_data' / 'tune_32768_k32_t20.txt', 'n': 32768, 'k': 32, 'T': 20},
]

MODES = [
    ('knn', 'knn.txt'),
    ('approx_knn', 'approx_knn.txt'),
    ('kmeans', 'kmeans.txt'),
]


def run_timed(cmd, cwd, env=None):
    start = time.perf_counter()
    subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return (time.perf_counter() - start) * 1000.0


def mae(path_a: Path, path_b: Path) -> float:
    out = subprocess.check_output(['python3', str(MAE_SCRIPT), str(path_a), str(path_b)], text=True)
    return float(out.strip().split()[-1])


def file_equal(path_a: Path, path_b: Path) -> bool:
    return subprocess.run(['diff', '-u', str(path_a), str(path_b)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def main():
    if TMP_ROOT.exists():
        shutil.rmtree(TMP_ROOT)
    TMP_ROOT.mkdir()

    env_gpu = os.environ.copy()
    env_gpu['OMP_NUM_THREADS'] = str(GPU_HOST_THREADS)

    rows = []
    for case in CASES:
        case_root = TMP_ROOT / case['name']
        seq_dir = case_root / 'seq'
        gpu_dir = case_root / 'gpu'
        seq_dir.mkdir(parents=True)
        gpu_dir.mkdir(parents=True)

        measurements = {}
        for mode, outfile in MODES:
            measurements[f'seq_{mode}_ms'] = run_timed([str(SEQ_BIN), str(case['path']), mode], seq_dir)
            measurements[f'gpu_{mode}_ms'] = run_timed([str(A2_BIN), str(case['path']), mode], gpu_dir, env_gpu)

        measurements['knn_outputs_match'] = int(file_equal(seq_dir / 'knn.txt', gpu_dir / 'knn.txt'))
        measurements['kmeans_outputs_match'] = int(file_equal(seq_dir / 'kmeans.txt', gpu_dir / 'kmeans.txt'))
        measurements['seq_approx_mae_vs_seq_knn'] = mae(seq_dir / 'knn.txt', seq_dir / 'approx_knn.txt')
        measurements['gpu_approx_mae_vs_gpu_knn'] = mae(gpu_dir / 'knn.txt', gpu_dir / 'approx_knn.txt')
        measurements['knn_speedup_vs_seq'] = measurements['seq_knn_ms'] / measurements['gpu_knn_ms']
        measurements['approx_speedup_vs_seq'] = measurements['seq_approx_knn_ms'] / measurements['gpu_approx_knn_ms']
        measurements['kmeans_speedup_vs_seq'] = measurements['seq_kmeans_ms'] / measurements['gpu_kmeans_ms']

        row = {
            'case': case['name'],
            'n': case['n'],
            'k': case['k'],
            'T': case['T'],
            **measurements,
        }
        rows.append(row)
        print(case['name'], row)

    fieldnames = [
        'case', 'n', 'k', 'T',
        'seq_knn_ms', 'gpu_knn_ms', 'knn_speedup_vs_seq', 'knn_outputs_match',
        'seq_approx_knn_ms', 'gpu_approx_knn_ms', 'approx_speedup_vs_seq',
        'seq_approx_mae_vs_seq_knn', 'gpu_approx_mae_vs_gpu_knn',
        'seq_kmeans_ms', 'gpu_kmeans_ms', 'kmeans_speedup_vs_seq', 'kmeans_outputs_match',
    ]
    with OUTPUT_CSV.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f'Wrote {OUTPUT_CSV}')


if __name__ == '__main__':
    main()
