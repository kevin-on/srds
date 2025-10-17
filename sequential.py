import argparse
import os
from datetime import datetime

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image

from utils.logger import log_info, setup_logging
from utils.utils import save_images_as_grid


def set_seed(seed: int):
    """Set random seed for reproducibility"""
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
    parser = argparse.ArgumentParser(description="Run sequential DDIM diffusion sampling")

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
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps for DDIM",
    )

    # Optional arguments
    parser.add_argument(
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-2",
        help="Hugging Face model ID",
    )
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="classifier-free guidance")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--log-file", type=str, default="log.txt", help="Log file path")
    
    return parser.parse_args()


def create_main_subdir(base_output_dir, timestamp, args):
    """Create the main subdirectory for this run"""
    subdir_name = f"{timestamp}_sequential_ddim{args.steps}"
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


def run_sequential_sampling(
    prompts,
    num_inference_steps,
    model_id,
    guidance_scale,
    height,
    width,
    generator,
    output_dir,
    logger
):
    """Run sequential DDIM sampling for given prompts"""
    
    log_info("Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")
    
    # Use DDIM scheduler
    pipe.scheduler = DDIMScheduler.from_pretrained(
        model_id, subfolder="scheduler", timestep_spacing="trailing"
    )
    
    log_info(f"Running sequential DDIM sampling with {num_inference_steps} steps...")
    
    for prompt_idx, prompt in enumerate(prompts):
        log_info(f"Processing prompt {prompt_idx + 1}/{len(prompts)}: {prompt}")
        
        # Generate image using sequential DDIM
        with torch.no_grad():
            images = pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
            ).images
        
        # Save individual image
        for img_idx, img in enumerate(images):
            img.save(os.path.join(output_dir, f"sequential_image_{prompt_idx}_{img_idx}.png"))
        
        log_info(f"Completed prompt {prompt_idx + 1}/{len(prompts)}")
    
    
    return all_images


if __name__ == "__main__":
    args = parse_args()

    # Setup basic output directory
    os.makedirs(args.output_dir, exist_ok=True)

    prompts = parse_prompts(args.prompts)

    set_seed(args.seed)
    generator = torch.Generator("cuda").manual_seed(args.seed)

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create main subdirectory for this run
    main_output_dir = create_main_subdir(args.output_dir, timestamp, args)
    
    log_file_path = os.path.join(main_output_dir, args.log_file)
    logger = setup_logging(log_file_path)
    log_info(f"Main output directory: {main_output_dir}")

    # Log execution parameters
    log_info("Sequential DDIM Execution Started")
    log_info(f"Output directory: {args.output_dir}")
    log_info(f"Prompts: {args.prompts}")
    log_info(f"Number of steps: {args.steps}")
    log_info(f"Seed: {args.seed}")
    log_info(f"Model: {args.model}")
    log_info(f"Guidance scale: {args.guidance_scale}")
    log_info(f"Height: {args.height}")
    log_info(f"Width: {args.width}")
    log_info(f"Loaded {len(prompts)} prompts")

    # Run sequential sampling for each prompt separately
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
            "algorithm": "sequential",
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "model": args.model,
            "seed": args.seed,
            "timestamp": timestamp,
            "height": args.height,
            "width": args.width
        }
        
        with open(os.path.join(prompt_output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Run sequential sampling for this single prompt
        images = run_sequential_sampling(
            prompts=[prompt],  # Single prompt
            num_inference_steps=args.steps,
            model_id=args.model,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            generator=generator,
            output_dir=prompt_output_dir,
            logger=logger
        )
        
        all_images.extend(images)
        log_info(f"Completed prompt {prompt_idx + 1}/{len(prompts)}")

    log_info("All prompts processed successfully!")
