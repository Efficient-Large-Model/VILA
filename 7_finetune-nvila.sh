# Optimized for 48GB GPU
export DEFAULT_RUN_NAME="NVILA-2B-finetune"
export DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=16
export DEFAULT_GRADIENT_ACCUMULATION_STEPS=4
export FP16=true
export DEFAULT_GPUS_PER_NODE=1

# ✅ Explicit W&B settings
export WANDB_ENTITY="fyp-21"
export WANDB_PROJECT="NVILA"
export WANDB_NAME="qved-nvila-finetune-$(date +%Y%m%d_%H%M%S)"

# Validation (optional)
export EVAL_DATA_MIXTURE="QVED-dataset-val"
export EVAL_STEPS=50
export SAVE_STEPS=100
export EVALUATION_STRATEGY="steps"

# Train
bash scripts/NVILA-Lite/sft.sh \
    Efficient-Large-Model/NVILA-Lite-2B \
    QVED-dataset \
    runs/train/nvila-2b-exercise-finetune \
    --fp16

echo "Training completed. Merging LoRA and uploading to Hugging Face..."
bash scripts/utils/merge_and_upload.sh
