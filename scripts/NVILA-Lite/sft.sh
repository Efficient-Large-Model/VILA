#!/bin/bash

DEFAULT_RUN_NAME=${DEFAULT_RUN_NAME:-"NVILA-2B-sft"}
DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=${DEFAULT_GLOBAL_TRAIN_BATCH_SIZE:-2048}
DEFAULT_GRADIENT_ACCUMULATION_STEPS=${DEFAULT_GRADIENT_ACCUMULATION_STEPS:-2}

STAGE_PATH=${1:-"runs/train/nvila-8b-pretrain/model"}
DATA_MIXTURE=${2:-"nvila-pretrain"}
OUTPUT_DIR=${3:-"runs/train/nvila-8b-sft"}

source scripts/setups/train.sh

if [ ! -z "$DEFAULT_GPUS_PER_NODE" ]; then
    GPUS_PER_NODE=$DEFAULT_GPUS_PER_NODE
    PER_DEVICE_TRAIN_BATCH_SIZE=$((GLOBAL_TRAIN_BATCH_SIZE / NNODES / GPUS_PER_NODE / GRADIENT_ACCUMULATION_STEPS))
fi

torchrun \
  --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $STAGE_PATH \
    --data_mixture $DATA_MIXTURE \
    --vision_tower Efficient-Large-Model/paligemma-siglip-so400m-patch14-448 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample_3x3_fix \
    --tune_vision_tower False \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio dynamic \
    --bf16 True \
    --output_dir $OUTPUT_DIR/model \
    --num_train_epochs 1 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 3e-5 \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lora_enable True \
    --lora_llm True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --report_to wandb
