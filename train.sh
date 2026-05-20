
set -euo pipefail

source env.sh
python train.py --arch "$MODEL_NAME" \
                --batch_size 2 \
                --gpu 0 \
                --train_ps 256 \
                --lr_initial 0.0001 \
                --save_dir "${SAVE_DIR}" \
                --warmup \
                --resume \
                --pretrain_weights "${CKPT_PATH}" \
                --train_dir "${TRAIN_DIR}" \
                --test_dir "${TEST_DIR}" \
                --save_images \
                --nepoch 91 \
                --checkpoint 10