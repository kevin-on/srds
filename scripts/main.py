import argparse
import os
import sys
from datetime import datetime

import torch

# Add parent directory to path so we can import src and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sparareal import StochasticParareal
from src.srds import SRDS
from utils.logger import setup_logging, log_info


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_prompts(prompts_input):
    """Parse prompts from text file (one prompt per line) or return single prompt"""
    if os.path.isfile(prompts_input):
        # Input is a file path
        with open(prompts_input, "r") as f:
            return [line.strip() for line in f if line.strip()]
    else:
        # Input is a direct prompt string
        return [prompts_input.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Run SRDS diffusion algorithm")

    # Required arguments
    parser.add_argument(
        "--prompts",
        "-p",
        type=str,
        help="Path to text file (one prompt per line) or direct prompt string",
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

    # Algorithm selection
    parser.add_argument(
        "--algorithm",
        "-a",
        type=str,
        choices=["srds", "sparareal"],
        default="srds",
        help="Algorithm to use: srds or sparareal",
    )
    parser.add_argument(
        "--num-samples",
        "-ns",
        type=int,
        default=3,
        help="Number of samples for sparareal algorithm (ignored for srds)",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="Stochasticity",
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
    parser.add_argument(
        "--log-file",
        type=str,
        default="log.txt",
        help="Log file path (default: log.txt)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Setup basic output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    prompts = parse_prompts(args.prompts)

    set_seed(args.seed)
    generator = torch.Generator("cuda").manual_seed(args.seed)

    # Create timestamped subdirectory with parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.algorithm == "sparareal":
        subdir_name = f"{timestamp}_cs{args.coarse_steps}-fs{args.fine_steps}_ns{args.num_samples}_eta{args.eta}"
    else:
        subdir_name = f"{timestamp}_cs{args.coarse_steps}-fs{args.fine_steps}"
    
    timestamped_output_dir = os.path.join(args.output_dir, subdir_name)
    os.makedirs(timestamped_output_dir, exist_ok=True)
    
    # Update log file path to use timestamped directory
    log_file_path = os.path.join(timestamped_output_dir, args.log_file)
    logger = setup_logging(log_file_path)
    
    # Log execution parameters
    log_info("SRDS/SParareal Execution Started")
    log_info(f"Algorithm: {args.algorithm}")
    log_info(f"Output directory: {args.output_dir}")
    log_info(f"Timestamped output directory: {timestamped_output_dir}")
    log_info(f"Log file: {log_file_path}")
    log_info(f"Prompts file: {args.prompts}")
    log_info(f"Coarse steps: {args.coarse_steps}")
    log_info(f"Fine steps: {args.fine_steps}")
    log_info(f"Tolerance: {args.tolerance}")
    log_info(f"Seed: {args.seed}")
    log_info(f"Model: {args.model}")
    if args.algorithm == "sparareal":
        log_info(f"Num samples: {args.num_samples}")
        log_info(f"Eta: {args.eta}")
    
    log_info(f"Loaded {len(prompts)} prompts")

    # Run selected algorithm
    if args.algorithm == "srds":
        log_info("Initializing SRDS algorithm...")
        algorithm = SRDS(model_id=args.model)
        log_info("Running SRDS...")
        algorithm(
            prompts=prompts,
            coarse_num_inference_steps=args.coarse_steps,
            fine_num_inference_steps=args.fine_steps,
            tolerance=args.tolerance,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            generator=generator,
            output_dir=timestamped_output_dir,
        )
    elif args.algorithm == "sparareal":
        log_info("Initializing SParareal algorithm...")
        algorithm = StochasticParareal(model_id=args.model)
        log_info("Running SParareal...")
        algorithm(
            prompts=prompts,
            coarse_num_inference_steps=args.coarse_steps,
            fine_num_inference_steps=args.fine_steps,
            num_samples=args.num_samples,
            tolerance=args.tolerance,
            guidance_scale=args.guidance_scale,
            eta=args.eta,
            height=args.height,
            width=args.width,
            generator=generator,
            output_dir=timestamped_output_dir,
        )
    
    log_info("Execution completed successfully!")
