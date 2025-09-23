import argparse
import os

import torch

from srds import SRDS


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_prompts(prompts_input):
    """Parse prompts from text file (one prompt per line)"""
    if not os.path.isfile(prompts_input):
        raise ValueError(f"Prompts file not found: {prompts_input}")

    with open(prompts_input, "r") as f:
        return [line.strip() for line in f if line.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Run SRDS diffusion algorithm")

    # Required arguments
    parser.add_argument(
        "--prompts",
        "-p",
        type=str,
        help="Path to text file (one prompt per line)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        required=True,
        help="Output directory for results",
    )

    # Step arguments
    parser.add_argument(
        "--coarse-steps",
        "-cs",
        type=int,
        default=10,
        help="Number of coarse inference steps",
    )
    parser.add_argument(
        "--fine-steps",
        "-fs",
        type=int,
        default=100,
        help="Number of fine inference steps",
    )
    parser.add_argument(
        "--tolerance", "-tol", type=float, default=0.1, help="Convergence tolerance"
    )

    # Optional arguments
    parser.add_argument(
        "--guidance-scale",
        "-gs",
        type=float,
        default=7.5,
        help="Guidance scale for classifier-free guidance",
    )
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-2",
        help="Hugging Face model ID",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prompts = parse_prompts(args.prompts)

    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    generator = torch.Generator("cuda").manual_seed(args.seed)

    # Initialize SRDS with model configuration
    srds = SRDS(model_id=args.model)

    # Run the algorithm
    srds(
        prompts=prompts,
        coarse_num_inference_steps=args.coarse_steps,
        fine_num_inference_steps=args.fine_steps,
        tolerance=args.tolerance,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        generator=generator,
        output_dir=args.output_dir,
    )
