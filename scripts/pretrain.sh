#!/bin/bash
# DINO MRI Pretraining Launch Script
#
# Run from project root:
#   bash scripts/pretrain.sh
#   NUM_GPUS=4 bash scripts/pretrain.sh
#   DATA_DIR=/abs/path/to/IXI-T1 bash scripts/pretrain.sh

set -e

NUM_GPUS=${NUM_GPUS:-1}
DATA_DIR=${DATA_DIR:-data/IXI-T1}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/dino_ixi}
CONFIG_FILE=${CONFIG_FILE:-configs/pretrain.yaml}

echo "========================================"
echo "DINO MRI Pretraining"
echo "  GPUs:       ${NUM_GPUS}"
echo "  Data:       ${DATA_DIR}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  Config:     ${CONFIG_FILE}"
echo "========================================"

if [ "$NUM_GPUS" -gt 1 ]; then
    accelerate launch \
        --num_processes=${NUM_GPUS} \
        --mixed_precision=fp16 \
        train.py \
        --config_file ${CONFIG_FILE} \
        --output_dir ${OUTPUT_DIR} \
        train.data_dir=${DATA_DIR}
else
    python train.py \
        --config_file ${CONFIG_FILE} \
        --output_dir ${OUTPUT_DIR} \
        train.data_dir=${DATA_DIR}
fi
