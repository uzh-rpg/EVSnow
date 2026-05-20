
set -euo pipefail

source env.sh
RUN_EVAL="${RUN_EVAL:-1}"

echo "MODEL_NAME=$MODEL_NAME"
for SUBFOLDER in "$SLIDER_DATASET_DIR"*/; do
    SUBFOLDER_NAME=$(basename "$SUBFOLDER")
    OUTPUT_PATH="${SLIDER_OUTPUT_DIR%/}/${SUBFOLDER_NAME}/"
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
        --dataset_folder "$SLIDER_OUTPUT_DIR" \
        --groundtruth_folder "$SLIDER_OUTPUT_DIR" \
        --model_name "$MODEL_NAME" \
        --visualize
fi