# Self-Refining Diffusion Samplers (SRDS)

Unofficial PyTorch implementation of
["Self-Refining Diffusion Samplers: Enabling Parallelization via Parareal Iterations"](https://arxiv.org/abs/2412.08292),
with additional experiments around adaptive and stochastic Parareal variants for text-to-image diffusion.

This repository started from reproducing SRDS without an official code release and then grew into a research codebase for:

- baseline `SRDS` reproduction on top of Stable Diffusion + DDIM
- `AdaptiveParareal` with a scheduled fine-solver budget
- `StochasticParareal` with multiple candidate trajectory generation strategies
- `SPararealTTS`, a reward-guided variant using PickScore
- experiment logging, sweeps, convergence analysis, and lightweight result visualization

The code is research-oriented rather than a polished package. The goal here is to make the implemented ideas and experiment flow explicit.

## Paper

- Paper: [Self-Refining Diffusion Samplers: Enabling Parallelization via Parareal Iterations](https://arxiv.org/abs/2412.08292)
- Original authors: Nikil Roashan Selvam, Amil Merchant, Stefano Ermon

## What I Implemented Here

Beyond a straight paper reproduction, this repo includes:

1. A full PyTorch reproduction of `SRDS` in `src/srds.py`, using Stable Diffusion and DDIM schedulers.
2. An `AdaptiveParareal` variant in `src/adaptiveparareal.py` that supports fine-step schedules such as `32x1,64x2,256x8`.
3. A `StochasticParareal` variant in `src/sparareal.py` that samples multiple candidate trajectory segments and selects among them.
4. A reward-guided variant in `src/spararealtts.py` that uses PickScore for final-step selection.
5. Supporting experiment infrastructure:
   - prompt-file or single-prompt execution
   - timestamped output folders and per-prompt configs
   - parameter sweeps
   - sequential DDIM baseline
   - convergence plots, HTML galleries, and simple local serving
6. Docker and VESSL-oriented workflow for running large batches on multiple GPUs.

## Repository Layout

```text
srds/
├── main.py                     # Main entry point for SRDS / SParareal / AdaptiveParareal / SPararealTTS
├── sequential.py               # Sequential DDIM baseline
├── sweep.py                    # Parameter sweep runner
├── src/
│   ├── srds.py                 # SRDS reproduction
│   ├── adaptiveparareal.py     # Adaptive Parareal extension
│   ├── sparareal.py            # Stochastic Parareal extension
│   ├── spararealtts.py         # Reward-guided stochastic Parareal
│   ├── diffusion.py            # Shared diffusion step utilities
│   └── reward/
│       ├── pickscore.py        # PickScore scorer
│       └── clip_score.py       # CLIP-based scorer
├── utils/
│   ├── logger.py               # Console/file logging
│   ├── metrics.py              # Sweep/result summaries
│   └── utils.py                # Image grids, prompt parsing helpers
├── evals/
│   ├── analyze_convergence.py  # Iteration-wise convergence analysis
│   ├── image_quality.py        # PickScore / CLIP-based image scoring
│   ├── generate_html_gallery.py # HTML gallery generation
│   └── serve_local.py          # Simple local HTTP server for galleries
├── prompts/
│   ├── 1.txt
│   ├── 3.txt
│   ├── 10.txt
│   └── 100.txt
├── requirements.txt
├── dev-requirements.txt
├── Dockerfile
└── pyproject.toml
```

## Setup

### Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dependencies

Core dependencies are:

- `torch`
- `diffusers`
- `transformers`
- `accelerate`
- `xformers`
- `pandas`
- `matplotlib`

The default backend in the current scripts is `stabilityai/stable-diffusion-2`.

## Running Experiments

The main entry point is `main.py`. `--prompts` can be either:

- a text file with one prompt per line, or
- a single prompt string

### 1. SRDS reproduction

```bash
python main.py \
  -p prompts/1.txt \
  -o output/srds \
  -a srds \
  -cs 16 \
  -fs 256
```

Optional experimental initialization:

```bash
python main.py \
  -p prompts/1.txt \
  -o output/srds_init \
  -a srds \
  -cs 16 \
  -fs 256 \
  -is 64
```

### 2. Adaptive Parareal

`AdaptiveParareal` takes a fine-step schedule of the form `steps x iterations`.

```bash
python main.py \
  -p prompts/3.txt \
  -o output/adaptive \
  -a adaptive \
  -cs 16 \
  -fss "32x2,64x2,256x8"
```

### 3. Stochastic Parareal

The stochastic variant generates multiple candidates per coarse segment.

DDIM-style stochasticity:

```bash
python main.py \
  -p prompts/3.txt \
  -o output/sparareal_ddim \
  -a sparareal \
  -cs 16 \
  -fs 256 \
  -ns 5 \
  -st "ddim,eta=0.5"
```

Directional perturbation:

```bash
python main.py \
  -p prompts/3.txt \
  -o output/sparareal_dir \
  -a sparareal \
  -cs 16 \
  -fs 256 \
  -ns 5 \
  -st "dir,scale=0.2"
```

### 4. Reward-guided SParareal

```bash
python main.py \
  -p prompts/1.txt \
  -o output/sparatts \
  -a sparatts \
  -cs 16 \
  -fs 256 \
  -ns 5 \
  -st "ddim,eta=0.5" \
  -rs pickscore
```

### 5. Sequential DDIM baseline

```bash
python sequential.py \
  -p prompts/1.txt \
  -o output/sequential \
  --steps 256
```

## Parameter Sweeps

`sweep.py` runs batched experiments for `srds` or `sparareal`.

```bash
python sweep.py \
  --prompt "a cinematic photo of a red sports car at sunset" \
  -o output/sweeps \
  -a sparareal \
  --coarse-steps 16 \
  --fine-steps 128 256 \
  --num-samples 3 5 \
  --sample-type "ddim,eta=0.1" "ddim,eta=0.5" \
  --tolerance 0.05 0.1
```

## Output Structure

Each run creates a timestamped experiment directory and then one subdirectory per prompt.

Typical contents of each prompt folder include:

```text
prompt0_example_prompt/
├── config.json
├── log.txt
├── srds_ddim_coarse.png
├── srds_initialized.png
├── srds_iteration_0.png
├── ...
├── srds_final.png
├── srds_ddim_gt.png
├── _gt_trajectory_errors_table.csv
├── _gt_trajectory_errors_plot.png
├── _trajectory_convergences_table.csv
└── _trajectory_convergences_plot.png
```

Adaptive Parareal uses `adaptive_parareal_*` filenames instead of `srds_*`.

## Analysis Utilities

### Convergence analysis

```bash
python evals/analyze_convergence.py --folder output
```

### Image scoring across iterations

PickScore:

```bash
python evals/image_quality.py --folder output --metric pickscore
```

CLIP:

```bash
python evals/image_quality.py --folder output --metric clip
```

### HTML gallery generation

```bash
python evals/generate_html_gallery.py --folder output
```

### Serve results locally

```bash
python evals/serve_local.py output --port 8000 --open
```

## Docker / VESSL Workflow

This repo also includes a Docker/VESSL-oriented workflow for larger runs.

- Docker image: `kevin0n/srds:latest`
- Dockerfile: `Dockerfile`

Example VESSL command block:

```yaml
run:
  - command: |-
      CUDA_VISIBLE_DEVICES=0 python main.py -p prompts/3.txt -o /root/output/adaptive -a adaptive -cs 16 -fss "32x1,256x16" &
      CUDA_VISIBLE_DEVICES=1 python main.py -p prompts/3.txt -o /root/output/adaptive -a adaptive -cs 16 -fss "32x2,256x16" &
      CUDA_VISIBLE_DEVICES=2 python main.py -p prompts/3.txt -o /root/output/adaptive -a adaptive -cs 16 -fss "32x4,256x16" &
      CUDA_VISIBLE_DEVICES=3 python main.py -p prompts/3.txt -o /root/output/adaptive -a adaptive -cs 16 -fss "32x8,256x16" &
      wait
    workdir: /root/code
```

## Notes

- This repository is not the official SRDS release.
- Several scripts are exploratory research code and were written for running/inspecting experiments quickly.
- The implementation is centered around text-to-image diffusion experiments rather than a general Parareal library.

## Citation

If you use this repository, please cite the original SRDS paper:

```bibtex
@article{selvam2024self,
  title={Self-Refining Diffusion Samplers: Enabling Parallelization via Parareal Iterations},
  author={Selvam, Nikil Roashan Selvam and Merchant, Amil and Ermon, Stefano},
  journal={arXiv preprint arXiv:2412.08292},
  year={2024}
}
```
