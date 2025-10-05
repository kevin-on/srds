#!/bin/bash

# Example parameter sweep for SParareal
echo "Starting parameter sweep for SParareal..."

# Basic sweep with different parameter combinations
python scripts/sweep.py \
    --prompt "a beautiful landscape with mountains and a lake" \
    --output-dir "./output/sweeps" \
    --algorithm sparareal \
    --coarse-steps 30 \
    --fine-steps 900 \
    --num-samples 1 5 10 \
    --sample-type "ddim,eta=0.01" "ddim,eta=0.005" \
    --tolerance 0.1

echo "Sweep completed!"
