#!/bin/bash

# Example parameter sweep for SParareal
echo "Starting parameter sweep for SParareal..."

# Basic sweep with different parameter combinations
CUDA_VISIBLE_DEVICES=1 python scripts/sweep.py \
    --prompt "a beautiful landscape with mountains and a lake" \
    --output-dir "./output/sweeps" \
    --algorithm sparareal \
    --coarse-steps 10 \
    --fine-steps 100 \
    --num-samples 1 5 10 \
    --sample-type "dir,scale=0.05" "dir,scale=0.1" "dir,eta=0.2" \
    --tolerance 0.1

echo "Sweep completed!"
