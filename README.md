# Self-Refining Diffusion Samplers (SRDS)

**Unofficial PyTorch Implementation of "Self-Refining Diffusion Samplers: Enabling Parallelization via Parareal Iterations"**

This repository provides an unofficial implementation of the SRDS algorithm described in the NeurIPS 2024 paper by Nikil Roashan Selvam, Amil Merchant, and Stefano Ermon.

📄 **Paper**: [Self-Refining Diffusion Samplers: Enabling Parallelization via Parareal Iterations](https://arxiv.org/abs/2412.08292)

## Usage

### Basic Usage

Run SRDS with a text file containing prompts (one per line):

```bash
python main.py --prompts prompts.txt --output-dir output/my_experiment
```

### Command Line Arguments

#### Required Arguments
- `--prompts`, `-p`: Path to text file with prompts (one per line)
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

- **`srds.py`**: Main SRDS algorithm implementation
- **`diffusion.py`**: Low-level diffusion step functions with batched/sequential variants
- **`utils.py`**: Utility functions for image processing and visualization
- **`main.py`**: Command-line interface and experiment runner

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
