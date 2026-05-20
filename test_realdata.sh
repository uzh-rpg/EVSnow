
set -euo pipefail

source env.sh
echo "MODEL_NAME=$MODEL_NAME"
echo "REALDATA_DATASET_DIR=$REALDATA_DATASET_DIR"
for SUBFOLDER in "$REALDATA_DATASET_DIR"*/; do
    SUBFOLDER_NAME=$(basename "$SUBFOLDER")
    echo "Processing subfolder: $SUBFOLDER_NAME"
    # only test on seq with folder name "14_02_05"
    if [ "$SUBFOLDER_NAME" != "14_02_05" ]; then
        continue
    fi
    OUTPUT_PATH="${REALDATA_OUTPUT_DIR%/}/${SUBFOLDER_NAME}/"
    echo "$SUBFOLDER"
    echo "$OUTPUT_PATH"
    python test.py --input_dir "$SUBFOLDER" \
                --result_dir "$OUTPUT_PATH"  \
                --model_name "$MODEL_NAME" \
                --weights "$CKPT_PATH" \
                --batch_size 1 \
                --skip_first_frames 60
done