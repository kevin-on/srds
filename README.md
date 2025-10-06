# Self-Refining Diffusion Samplers (SRDS)

**Unofficial PyTorch Implementation of "Self-Refining Diffusion Samplers: Enabling Parallelization via Parareal Iterations"**

This repository provides an unofficial implementation of the SRDS algorithm described in the NeurIPS 2024 paper by Nikil Roashan Selvam, Amil Merchant, and Stefano Ermon.

📄 **Paper**: [Self-Refining Diffusion Samplers: Enabling Parallelization via Parareal Iterations](https://arxiv.org/abs/2412.08292)

## Project Structure

```
srds/
├── src/                    # Core algorithms
│   ├── srds.py            # SRDS algorithm implementation
│   ├── sparareal.py       # SParareal algorithm implementation
│   └── diffusion.py       # Diffusion step functions
├── scripts/                # Execution scripts
│   ├── main.py            # Main execution script
│   ├── sweep.py           # Parameter sweep script
│   └── run_sweep_example.sh # Sweep example
├── utils/                  # Utilities
│   ├── utils.py           # Image processing utilities
│   └── logger.py          # Logging utilities
└── output/                # Experiment results
```

## Usage

### Basic Usage

Run SRDS with a text file containing prompts (one per line):

```bash
python scripts/main.py --prompts example_prompts.txt --output-dir output/my_experiment
```

Or run with a single direct prompt:

```bash
python scripts/main.py --prompts "a beautiful landscape with mountains and a lake" --output-dir output/single_test
```

### Parameter Sweep

Run parameter sweep for a single prompt:

```bash
python scripts/sweep.py \
    --prompt "a beautiful landscape with mountains and a lake" \
    --output-dir "./output/sweeps" \
    --algorithm sparareal \
    --coarse-steps 5 10 15 \
    --fine-steps 50 100 150 \
    --num-samples 1 5 10 \
    --eta 0.01 0.1 1.0 \
    --tolerance 0.01 0.05 0.1
```

### Command Line Arguments

#### Required Arguments
- `--prompts`, `-p`: Path to text file with prompts (one per line) OR direct prompt string
- `--output-dir`, `-o`: Output directory for generated images and analysis

#### Algorithm Parameters
- `--coarse-steps`, `-cs`: Number of coarse inference steps (default: 10)
- `--fine-steps`, `-fs`: Number of fine inference steps (default: 100)
- `--tolerance`, `-tol`: Convergence tolerance (default: 0.1)

#### Generation Parameters  
- `--guidance-scale`, `-gs`: Classifier-free guidance scale (default: 7.5)
- `--height`: Image height in pixels (default: 512)
- `--width`: Image width in pixels (default: 512)
- `--seed`, `-s`: Random seed for reproducibility (default: 42)
- `--model`: Hugging Face model ID (default: "stabilityai/stable-diffusion-2")

## Output Structure

The algorithm generates comprehensive outputs for analysis:

```
output/
├── srds_ddim_course.png       # Coarse DDIM baseline
├── srds_ddim_gt.png           # Fine DDIM ground truth  
├── srds_initialized.png       # Initial trajectory
├── srds_iteration_*.png       # Results from each SRDS iteration
├── srds_final.png            # Final converged result
├── trajectory_errors_plot.png # Convergence visualization
└── trajectory_errors_table.csv # Numerical error data
```

## Implementation Details

### Core Components

- **`src/srds.py`**: Main SRDS algorithm implementation
- **`src/sparareal.py`**: SParareal algorithm implementation
- **`src/diffusion.py`**: Low-level diffusion step functions with batched/sequential variants
- **`utils/utils.py`**: Utility functions for image processing and visualization
- **`utils/logger.py`**: Logging utilities
- **`scripts/main.py`**: Command-line interface and experiment runner
- **`scripts/sweep.py`**: Parameter sweep functionality

### Code Formatting

This project uses [Ruff](https://docs.astral.sh/ruff/) for code formatting and linting. Before committing changes:

```bash
# Format code automatically
ruff format .

# Check for linting issues
ruff check .

# Auto-fix issues where possible
ruff check --fix .
```

Ruff is configured in `pyproject.toml` with a 100-character line length and targets Python 3.8+.

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{selvam2024self,
  title={Self-Refining Diffusion Samplers: Enabling Parallelization via Parareal Iterations},
  author={Selvam, Nikil Roashan Selvam and Merchant, Amil and Ermon, Stefano},
  journal={arXiv preprint arXiv:2412.08292},
  year={2024}
}
```

## License

This implementation is provided for research purposes. Please refer to the original paper and model licenses for usage terms.
