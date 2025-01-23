#!/bin/bash
DATA="Office"
CKPT="checkpoints/EARec-v1.5-7b-FHCKM-Office-image-cf90-text-lora"
Modal="TVB"
EMB_SUFFIX="featEARec.tvb"

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.get_modal_emb \
    --model-path ${CKPT} \
    --model-base liuhaotian/llava-v1.5-7b \
    --image-folder dataset/${DATA}/image \
    --question-file dataset/${DATA}/Item_Modal_${Modal}.json \
    --answers-file dataset/${DATA}/${DATA}.${EMB_SUFFIX} \
