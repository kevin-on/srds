import argparse
import os
from datetime import datetime

import torch


from src.srds import SRDS
from src.sparareal import StochasticParareal
from src.spararealtts import SPararealTTS

from utils.logger import log_info, setup_logging


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
        with open(prompts_input) as f:
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
        choices=["srds", "sparareal", "sparatts"],
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
        "--sample_type",
        "-st",
        type=str,
        default=None, #"ddim,eta=1.0",
        help="Method of adding stochasticity",
    )
    parser.add_argument(
        "--reward_scorer",
        "-rs",
        type=str,
        default=None,
        help="reward scorer for SPararealTTS algorithm"
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


def create_main_subdir(base_output_dir, timestamp, args):
    """Create the main subdirectory for this run"""
    if args.algorithm == "sparatts":
        subdir_name = f"{timestamp}_sparatts_cs{args.coarse_steps}-fs{args.fine_steps}_ns{args.num_samples}"
    elif args.algorithm == "sparareal":
        subdir_name = f"{timestamp}_sparareal_cs{args.coarse_steps}-fs{args.fine_steps}_{args.sample_type}-ns{args.num_samples}"
    elif args.algorithm == "srds":
        subdir_name = f"{timestamp}_srds_cs{args.coarse_steps}-fs{args.fine_steps}"
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    
    main_output_dir = os.path.join(base_output_dir, subdir_name)
    os.makedirs(main_output_dir, exist_ok=True)
    return main_output_dir


def create_prompt_subdir(main_output_dir, prompt_idx, prompt_text):
    """Create a subdirectory for each prompt within the main directory"""
    # 프롬프트를 파일명으로 사용할 수 있도록 정리 (최대 30자)
    safe_prompt = "".join(c for c in prompt_text[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_prompt = safe_prompt.replace(' ', '_')
    
    prompt_subdir_name = f"prompt{prompt_idx}_{safe_prompt}"
    prompt_output_dir = os.path.join(main_output_dir, prompt_subdir_name)
    os.makedirs(prompt_output_dir, exist_ok=True)
    
    return prompt_output_dir


if __name__ == "__main__":
    args = parse_args()

    # Setup basic output directory
    os.makedirs(args.output_dir, exist_ok=True)

    prompts = parse_prompts(args.prompts)

    set_seed(args.seed)
    generator = torch.Generator("cuda").manual_seed(args.seed)

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Log execution parameters
    log_info("SRDS/SParareal Execution Started")
    log_info(f"Algorithm: {args.algorithm}")
    log_info(f"Output directory: {args.output_dir}")
    log_info(f"Prompts: {args.prompts}")
    log_info(f"Coarse steps: {args.coarse_steps}")
    log_info(f"Fine steps: {args.fine_steps}")
    log_info(f"Tolerance: {args.tolerance}")
    log_info(f"Seed: {args.seed}")
    log_info(f"Model: {args.model}")

    if args.algorithm == "sparareal":
        log_info(f"Num samples: {args.num_samples}")
        log_info(f"Sample type: {args.sample_type}")

    if args.algorithm == "sparatts":
        log_info(f"Num samples: {args.num_samples}")
        log_info(f"Sample type: {args.sample_type}")
        log_info(f"Reward scorer: {args.reward_scorer}")

    log_info(f"Loaded {len(prompts)} prompts")

    # Create main subdirectory for this run
    main_output_dir = create_main_subdir(args.output_dir, timestamp, args)
    log_info(f"Main output directory: {main_output_dir}")

    # Run algorithm for each prompt separately
    all_images = []
    for prompt_idx, prompt in enumerate(prompts):
        log_info(f"Processing prompt {prompt_idx + 1}/{len(prompts)}: {prompt}")
        
        # Create prompt-specific subdirectory within main directory
        prompt_output_dir = create_prompt_subdir(main_output_dir, prompt_idx, prompt)
        
        # Update log file path for this prompt
        log_file_path = os.path.join(prompt_output_dir, args.log_file)
        logger = setup_logging(log_file_path)
        
        log_info(f"Processing prompt: {prompt}")
        log_info(f"Prompt output directory: {prompt_output_dir}")
        
        # 프롬프트 정보 저장
        import json
        config = {
            "prompt": prompt,
            "prompt_idx": prompt_idx,
            "algorithm": args.algorithm,
            "coarse_steps": args.coarse_steps,
            "fine_steps": args.fine_steps,
            "tolerance": args.tolerance,
            "guidance_scale": args.guidance_scale,
            "sample_type": args.sample_type,
            "num_samples": args.num_samples,
            "reward_scorer": args.reward_scorer,
            "model": args.model,
            "seed": args.seed,
            "timestamp": timestamp
        }
        
        with open(os.path.join(prompt_output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Run algorithm for this single prompt
        if args.algorithm == "srds":
            log_info("Initializing SRDS algorithm...")
            algorithm = SRDS(model_id=args.model)
            log_info("Running SRDS...")
            images = algorithm(
                prompts=[prompt],  # Single prompt
                coarse_num_inference_steps=args.coarse_steps,
                fine_num_inference_steps=args.fine_steps,
                tolerance=args.tolerance,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                generator=generator,
                output_dir=prompt_output_dir,
            )
        elif args.algorithm == "sparareal":
            if args.sample_type is None:
                raise ValueError("sample_type must be specified when using sparareal algorithm")
            
            log_info("Initializing SParareal algorithm...")
            algorithm = StochasticParareal(model_id=args.model)
            log_info("Running SParareal...")
            images = algorithm(
                prompts=[prompt],  # Single prompt
                coarse_num_inference_steps=args.coarse_steps,
                fine_num_inference_steps=args.fine_steps,
                num_samples=args.num_samples,
                tolerance=args.tolerance,
                guidance_scale=args.guidance_scale,
                sample_type=args.sample_type,
                height=args.height,
                width=args.width,
                generator=generator,
                output_dir=prompt_output_dir,
            )
        elif args.algorithm == "sparatts":
            if args.sample_type is None:
                raise ValueError("sample_type must be specified when using spararealtts algorithm")
            
            if args.reward_scorer is None:
                raise ValueError("reward_scorer must be specified when using spararealtts algorithm")
            

            log_info("Initializing SPararealTTS algorithm...")
            algorithm = SPararealTTS(model_id=args.model)
            log_info("Running SPararealTTS...")
            images = algorithm(
                prompts=[prompt],  # Single prompt
                coarse_num_inference_steps=args.coarse_steps,
                fine_num_inference_steps=args.fine_steps,
                num_samples=args.num_samples,
                tolerance=args.tolerance,
                guidance_scale=args.guidance_scale,
                sample_type=args.sample_type,
                height=args.height,
                width=args.width,
                generator=generator,
                output_dir=prompt_output_dir,
                reward_scorer=args.reward_scorer
            )
        else:
            raise ValueError(f"Unknown algorithm: {args.algorithm}")

        all_images.extend(images)
        log_info(f"Completed prompt {prompt_idx + 1}/{len(prompts)}")

    log_info("All prompts processed successfully!")
