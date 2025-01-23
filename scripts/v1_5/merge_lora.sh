#!/bin/bash
python scripts/merge_multi_lora.py \
                checkpoints/EARec-v1.5-7b-FHCKM-finetune_image_lora \
                checkpoints/EARec-v1.5-7b-FHCKM-Office-finetune_cf_lora \
                checkpoints/EARec-v1.5-7b-FHCKM-finetune_text_lora \
                -o checkpoints/EARec-v1.5-7b-FHCKM-Office-image-cf90-text-lora \
                --strategy weighted_sum --weights 0.05,0.9,0.05