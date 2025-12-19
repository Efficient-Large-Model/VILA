#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QVED_DIR="$SCRIPT_DIR/data_prepare/qved"

# Variables for all script arguments
DATASET_ROOT_DIR="llava/data/registry/datasets"
DATASET_DIR="$DATASET_ROOT_DIR/dataset"
FILTERED_JSON="$DATASET_ROOT_DIR/ground_truth.json"
CONVERTED_JSON="$DATASET_DIR/converted.json"
VIDEOS_DIR="$DATASET_DIR/videos"
TRAIN_JSON="$DATASET_DIR/train.json"
VAL_JSON="$DATASET_DIR/val.json"
TEST_JSON="$DATASET_DIR/test.json"

mkdir -p "$DATASET_DIR"
mkdir -p "$VIDEOS_DIR"

# 1. Load dataset
python "$QVED_DIR/load_dataset.py" "$DATASET_ROOT_DIR"

# 2. Filter ground truth
python "$QVED_DIR/filter_ground_truth.py" "$DATASET_ROOT_DIR"

# 3. Clean dataset (NEW)
python "$QVED_DIR/clean_videos.py" "$DATASET_ROOT_DIR"

# 4. Convert to chat template
python "$QVED_DIR/convert_to_chat_template.py" "$FILTERED_JSON" "$CONVERTED_JSON"

# 5. Copy data to videos directory
python "$QVED_DIR/copy_data_to_videos_dir.py" "$DATASET_ROOT_DIR" "$VIDEOS_DIR"

# 6. Split dataset
python "$QVED_DIR/split_dataset.py" "$CONVERTED_JSON" "$TRAIN_JSON" "$VAL_JSON" "$TEST_JSON"

echo "All dataset preparation steps completed."