
set -euo pipefail

PYTHON_EXE=preprocess.py
DATASET_DIR="${DATASET_DIR:-./dsec_snow_dataset}"
TEST_DIR="${TEST_DIR:-${DATASET_DIR%/}/test/}"

# Control the number of concurrent jobs
MAX_JOBS=2
current_jobs=0

for SUBFOLDER in "$TEST_DIR"*/; do

    echo python "$PYTHON_EXE" "$SUBFOLDER"
    python "$PYTHON_EXE" --seq "$SUBFOLDER" &

    ((current_jobs++))
    if [[ $current_jobs -ge $MAX_JOBS ]]; then
        wait -n
        ((current_jobs--))
    fi
done

# Wait for all background jobs to finish
wait

