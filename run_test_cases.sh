#!/bin/bash
set -euo pipefail

LOG_FILE="test_case_outputs.log"
>"$LOG_FILE"

case_list=(
    "1000 64 25"
    "1000 64 50"
    "1000 128 25"
    "1000 128 50"
    "10000 64 25"
    "10000 64 50"
    "10000 128 25"
    "10000 128 50"
    "100000 64 25"
    "100000 64 50"
    "100000 128 25"
    "100000 128 50"
)

for test_case in "${case_list[@]}"; do
    read -r n k T <<< "$test_case"

    {
        printf "\n=== Test case: n=%s k=%s T=%s ===\n" "$n" "$k" "$T"
        python3 dataset_generator.py --n "$n" --k "$k" --T "$T"
        ./run_all.sh
    } >>"$LOG_FILE" 2>&1

done

echo "Captured outputs for ${#case_list[@]} test cases in $LOG_FILE"
