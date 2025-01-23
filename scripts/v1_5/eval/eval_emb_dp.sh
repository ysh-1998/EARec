#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

DATA="Office"
CKPT="checkpoints/EARec-v1.5-7b-FHCKM-Office-image-cf90-text-lora"
Modal="TVB"
EMB_SUFFIX="featEARec.tvb"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.get_modal_emb \
        --model-path ${CKPT} \
        --model-base liuhaotian/llava-v1.5-7b \
        --image-folder dataset/${DATA}/image \
        --question-file dataset/${DATA}/Item_Modal_${Modal}.json \
        --answers-file  dataset/${DATA}/Emb_${EMB_SUFFIX}/${CHUNKS}_${IDX}.emb \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX  &
done
