
set -euo pipefail

source env.sh
RUN_EVAL="${RUN_EVAL:-1}"

for SUBFOLDER in "$TEST_DIR"*/; do
    SUBFOLDER_NAME=$(basename "$SUBFOLDER")
    OUTPUT_PATH="${DSEC_OUTPUT_DIR%/}/${SUBFOLDER_NAME}/"
    echo "$SUBFOLDER"
    echo "$OUTPUT_PATH"
    python test.py --input_dir "$SUBFOLDER" \
                --result_dir "$OUTPUT_PATH"  \
                --model_name "$MODEL_NAME" \
                --weights "$CKPT_PATH" \
                --batch_size 1 \
                --has_groundtruth
done

if [[ "$RUN_EVAL" == "1" ]]; then
    python evaluate.py \
        --dataset_folder "$DSEC_OUTPUT_DIR" \
        --groundtruth_folder "$DSEC_OUTPUT_DIR" \
        --model_name "$MODEL_NAME" \
        --visualize
fi