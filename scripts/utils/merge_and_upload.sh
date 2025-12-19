#!/bin/bash
set -e

BASE_MODEL="Efficient-Large-Model/NVILA-Lite-2B"
LORA_DIR="runs/train/nvila-2b-exercise-finetune/model"
MERGED_DIR="runs/train/nvila-2b-exercise-finetune/merged"
HF_REPO="EdgeVLM-Labs/NVILA-Lite-2B-QVED"

echo "Merging LoRA adapters..."

python - <<EOF
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "$BASE_MODEL", torch_dtype="auto", device_map="cpu"
)
model = PeftModel.from_pretrained(model, "$LORA_DIR")
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained("$BASE_MODEL")

model.save_pretrained("$MERGED_DIR", safe_serialization=True)
tokenizer.save_pretrained("$MERGED_DIR")
EOF

echo "Uploading merged model to Hugging Face..."
huggingface-cli upload "$HF_REPO" "$MERGED_DIR" --include="*"

echo "✅ Upload complete: https://huggingface.co/$HF_REPO"
