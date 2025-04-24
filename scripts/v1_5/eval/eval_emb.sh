#!/bin/bash
DATA="Office"
CKPT="checkpoints/EARec-v1.5-7b-FHCKM-Office-image-cf90-text-lora"
EMB_SUFFIX="featEARec.tvb"

CUDA_VISIBLE_DEVICES=0 python -m mllm.eval.get_modal_emb \
    --model-path ${CKPT} \
    --model-base liuhaotian/llava-v1.5-7b \
    --image-folder dataset/${DATA}/image \
    --question-file dataset/${DATA}/Item_Modal.json \
    --answers-file dataset/${DATA}/${DATA}.${EMB_SUFFIX} \
