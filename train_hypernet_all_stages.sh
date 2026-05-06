#!/bin/bash

# Multi-stage training script for Hypernet kernel
# Runs stage1 -> stage2 -> stage3 sequentially
# Each stage automatically loads the checkpoint from the previous stage

set -e  # Exit on error

echo "=========================================="
echo "Starting Multi-Stage Hypernet Training"
echo "=========================================="

# Configuration files
STAGE1_CONFIG="./configs/training_config_hypernet_stage1.yaml"
STAGE2_CONFIG="./configs/training_config_hypernet_stage2.yaml"
STAGE3_CONFIG="./configs/training_config_hypernet_stage3.yaml"
MAIN_CONFIG="./configs/training_config.yaml"

# Check if config files exist
for config in "$STAGE1_CONFIG" "$STAGE2_CONFIG" "$STAGE3_CONFIG"; do
    if [ ! -f "$config" ]; then
        echo "Error: Config file not found: $config"
        exit 1
    fi
done

# Optional: generate hydra_hypernet_mark.pt if missing (requires models/hydra_bert_23layers.pt)
if [ ! -f "./models/hydra_hypernet_mark.pt" ]; then
    echo "Creating hydra_hypernet_mark.pt via transfer..."
    cp "$STAGE1_CONFIG" "$MAIN_CONFIG" && python -m src.transfer
    echo ""
fi

# Function to run a training stage
run_stage() {
    local stage_num=$1
    local config_file=$2
    local stage_name=$3
    
    echo ""
    echo "=========================================="
    echo "Starting Stage $stage_num: $stage_name"
    echo "Config: $config_file"
    echo "=========================================="
    
    # Copy stage config to main config
    cp "$config_file" "$MAIN_CONFIG"
    
    # Run training
    python -m src.main
    
    if [ $? -ne 0 ]; then
        echo "Error: Stage $stage_num failed!"
        exit 1
    fi
    
    echo ""
    echo "Stage $stage_num completed successfully!"
    echo "Checkpoint saved in: ./checkpoints/hydra_mark"
    echo ""
}

# Stage 1: Pretraining
run_stage 1 "$STAGE1_CONFIG" "Pretraining"

# Backup stage1 checkpoint
if [ -d "./checkpoints/hydra_mark" ]; then
    cp -r ./checkpoints/hydra_mark ./checkpoints/hydra_mark_hypernet_stage1
    echo "Stage 1 checkpoint backed up to: ./checkpoints/hydra_mark_hypernet_stage1"
fi

# Stage 2: Intermediate Fine-tuning
run_stage 2 "$STAGE2_CONFIG" "Intermediate Fine-tuning"

# Backup stage2 checkpoint
if [ -d "./checkpoints/hydra_mark" ]; then
    cp -r ./checkpoints/hydra_mark ./checkpoints/hydra_mark_hypernet_stage2
    echo "Stage 2 checkpoint backed up to: ./checkpoints/hydra_mark_hypernet_stage2"
fi

# Stage 3: Task-specific Fine-tuning
run_stage 3 "$STAGE3_CONFIG" "Task-specific Fine-tuning"

# Backup stage3 checkpoint
if [ -d "./checkpoints/hydra_mark" ]; then
    cp -r ./checkpoints/hydra_mark ./checkpoints/hydra_mark_hypernet_stage3
    echo "Stage 3 checkpoint backed up to: ./checkpoints/hydra_mark_hypernet_stage3"
fi

echo ""
echo "=========================================="
echo "All Stages Completed Successfully!"
echo "=========================================="
echo ""
echo "Final checkpoints:"
echo "  - Stage 1: ./checkpoints/hydra_mark_hypernet_stage1"
echo "  - Stage 2: ./checkpoints/hydra_mark_hypernet_stage2"
echo "  - Stage 3: ./checkpoints/hydra_mark_hypernet_stage3"
echo "  - Latest:  ./checkpoints/hydra_mark"
echo ""
