#!/bin/bash
dataset=$1
run_name=$dataset-$2
root_dir="dataset"

deepspeed --include localhost:0,1,2,3 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version plain \
    --data_path ${root_dir}/${dataset}/ID_Text_train.json \
    --eval_data_path ${root_dir}/${dataset}/ID_Text_test.json \
    --cf_tower ${root_dir}/${dataset}/item_emb.pth \
    --mm_cf_projector_type mlp2x_gelu \
    --tune_mm_cf_mlp_adapter True \
    --bf16 True \
    --output_dir ./checkpoints/EARec-v1.5-7b-${run_name} \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $run_name
