# Optimized for 48GB GPU - can handle larger batch sizes
export DEFAULT_RUN_NAME="NVILA-2B-finetune"
export DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=16  
export DEFAULT_GRADIENT_ACCUMULATION_STEPS=4
export FP16=true
export DEFAULT_GPUS_PER_NODE=1

export WANDB_PROJECT="NVILA"
export WANDB_NAME="finetune-nvila-2b-qved"

# Add validation dataset support (optional)
export EVAL_DATA_MIXTURE="QVED-dataset-val"  # Validation dataset
export EVAL_STEPS=50  # Evaluate every 50 steps
export SAVE_STEPS=100  # Save every 100 steps
export EVALUATION_STRATEGY="steps"  # Enable evaluation

# Fine-tune NVILA-Lite-2B
bash scripts/NVILA-Lite/sft.sh \
    Efficient-Large-Model/NVILA-Lite-2B \
    QVED-dataset \
    runs/train/nvila-2b-exercise-finetune \
    --fp16

# Upload to Hugging Face after training
echo "Training completed. Uploading model to Hugging Face..."
huggingface-cli upload EdgeVLM-Labs/NVILA-Lite-2B-finetuned-500 runs/train/nvila-2b-exercise-finetune/model --include="*"
