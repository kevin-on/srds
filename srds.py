from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from diffusion import diffusion_step
from utils import decode_latents_to_pil, save_images_as_grid

"""
Implementation of Self-Refining Diffusion Samplers (SRDS) algorithm.

Future Enhancements:
- Implement parallel processing for fine-solver
- Implement pipelining
"""


@torch.no_grad()
def run_srds_diffusion(
    prompts: List[str],
    coarse_num_inference_steps: int,
    fine_num_inference_steps: int,
    tolerance: float,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
    generator: torch.Generator = None,
    model_id: str = "stabilityai/stable-diffusion-2",
    output_dir: str = "output",
):
    if fine_num_inference_steps % coarse_num_inference_steps != 0:
        raise ValueError(
            "The fine num inference steps must be a multiple of the coarse num inference steps"
        )

    scheduler_coarse = DDIMScheduler.from_pretrained(
        model_id, subfolder="scheduler", timestep_spacing="trailing"
    )
    scheduler_fine = DDIMScheduler.from_pretrained(
        model_id, subfolder="scheduler", timestep_spacing="trailing"
    )
    scheduler_coarse.set_timesteps(coarse_num_inference_steps)
    scheduler_fine.set_timesteps(fine_num_inference_steps)

    coarse_timesteps = scheduler_coarse.timesteps
    fine_timesteps = scheduler_fine.timesteps
    if not all(t in fine_timesteps for t in coarse_timesteps):
        raise ValueError("The coarse timesteps are not a subset of the fine timesteps")
    coarse_indices_tensor = torch.cat(
        [torch.where(fine_timesteps == t)[0] for t in coarse_timesteps]
    )

    # Timestep setup
    print("SRDS Algorithm Configuration:")
    print(f"  Coarse steps: {len(coarse_timesteps)} | timesteps: {coarse_timesteps}")
    print(f"  Fine steps: {len(fine_timesteps)} | timesteps: {fine_timesteps}")

    pipe_coarse = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler_coarse,
    )
    pipe_coarse = pipe_coarse.to("cuda")

    # Encode prompt
    do_classifier_free_guidance = guidance_scale > 1.0
    prompt_embeds, negative_prompt_embeds = pipe_coarse.encode_prompt(
        prompt=prompts,
        device=pipe_coarse.device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
    )

    # For classifier free guidance, concatenate the unconditional and text embeddings
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # Prepare latent variables
    num_channels_latents = pipe_coarse.unet.config.in_channels
    initial_latents = pipe_coarse.prepare_latents(  # line 1 of Algorithm 1
        len(prompts),
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        pipe_coarse.device,
        generator,
    )

    # Store ground truth DDIM trajectory in fine pipeline
    def compute_gt_trajectory():
        latents_gt_list = []
        latents_gt = initial_latents.clone()
        for i in range(fine_num_inference_steps):
            t = scheduler_fine.timesteps[i]
            if t in coarse_timesteps:
                latents_gt_list.append(latents_gt.clone())
            latents_gt = diffusion_step(
                latents_gt,
                t,
                prompt_embeds,
                pipe_coarse.unet,
                scheduler_fine,
                guidance_scale,
                do_classifier_free_guidance,
            )
        latents_gt_list.append(latents_gt.clone())
        gt_trajectory = torch.cat(latents_gt_list[1:], dim=0)

        return gt_trajectory

    gt_trajectory = compute_gt_trajectory()

    # Run SRDS diffusion
    prev_latents = torch.cat(
        [initial_latents] * coarse_num_inference_steps, dim=0
    )  # prev_{i} in Algorithm 1
    cur_latents = prev_latents.clone()  # cur_{i} in Algorithm 1

    # Initialize previous latents
    for i in range(coarse_num_inference_steps):  # line 2 of Algorithm 1
        current_slice = slice(i * len(prompts), (i + 1) * len(prompts))
        prev_slice = slice((i - 1) * len(prompts), i * len(prompts))

        prev_latents[current_slice] = diffusion_step(  # line 3 of Algorithm 1
            (prev_latents[prev_slice] if i > 0 else initial_latents),
            coarse_timesteps[i],
            prompt_embeds,
            pipe_coarse.unet,
            scheduler_coarse,
            guidance_scale,
            do_classifier_free_guidance,
        )

    x = prev_latents.clone()  # x_{i} in Algorithm 1, line 4 of Algorithm 1

    def sanity_check_initialization():
        """
        Sanity check initialized latents
        Initialized latents at t = 0 should be the same as the that of the DDIM coarse output
        """
        coarse_output = pipe_coarse(
            prompts,
            num_inference_steps=coarse_num_inference_steps,
            guidance_scale=guidance_scale,
            latents=initial_latents,
        )

        save_images_as_grid(coarse_output.images, f"{output_dir}/srds_ddim_coarse.png")

        initialized_images = decode_latents_to_pil(
            prev_latents[-len(prompts) :], pipe_coarse
        )
        save_images_as_grid(initialized_images, f"{output_dir}/srds_initialized.png")

        if len(coarse_output.images) == len(initialized_images) and all(
            coarse_img.tobytes() == init_img.tobytes()
            for coarse_img, init_img in zip(coarse_output.images, initialized_images)
        ):
            print(
                "PASS: Initialization verified - generated image matches DDIM coarse baseline"
            )
        else:
            print(
                "FAIL: Initialization error - generated image differs from DDIM baseline"
            )

    sanity_check_initialization()

    trajectory_errors = []
    prev_iter_images: Image.Image = None

    for srds_iter in tqdm(
        range(coarse_num_inference_steps), desc="SRDS Iterations"
    ):  # line 6 of Algorithm 1

        y = torch.cat(
            [initial_latents, x[: -len(prompts)]], dim=0
        )  # y_{i} in Algorithm 1

        # Update description for fine steps
        tqdm.write(
            f"SRDS Iteration {srds_iter+1}/{coarse_num_inference_steps} - Processing fine steps"
        )
        for i in range(coarse_num_inference_steps):  # line 7 of Algorithm 1
            current_slice = slice(i * len(prompts), (i + 1) * len(prompts))
            timestep_start = coarse_timesteps[i]
            timestep_end = (
                coarse_timesteps[i + 1] if i < coarse_num_inference_steps - 1 else -1
            )
            y[current_slice] = diffusion_step(
                latents=y[current_slice],
                timestep=timestep_start,
                timestep_end=timestep_end,
                prompt_embeds=prompt_embeds,
                unet=pipe_coarse.unet,
                scheduler=scheduler_fine,
                guidance_scale=guidance_scale,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )

        # Update description for coarse steps
        tqdm.write(
            f"SRDS Iteration {srds_iter+1}/{coarse_num_inference_steps} - Processing coarse sweep"
        )
        for i in range(coarse_num_inference_steps):  # line 9 of Algorithm 1
            current_slice = slice(i * len(prompts), (i + 1) * len(prompts))
            prev_slice = slice((i - 1) * len(prompts), i * len(prompts))

            cur_latents[current_slice] = diffusion_step(  # line 10 of Algorithm 1
                (x[prev_slice] if i > 0 else initial_latents),
                coarse_timesteps[i],
                prompt_embeds,
                pipe_coarse.unet,
                scheduler_coarse,
                guidance_scale,
                do_classifier_free_guidance,
            )
            x[current_slice] = y[current_slice] + (
                cur_latents[current_slice] - prev_latents[current_slice]
            )  # line 11 of Algorithm 1
            prev_latents[current_slice] = cur_latents[
                current_slice
            ]  # line 12 of Algorithm 1

        diff_reshaped = rearrange(
            x - gt_trajectory,
            "(steps prompts) ... -> steps prompts ...",
            steps=coarse_num_inference_steps,
            prompts=len(prompts),
        )
        trajectory_errors.append(torch.norm(diff_reshaped.flatten(1), dim=1))

        # Save final images of every iteration
        images = decode_latents_to_pil(x[-len(prompts) :], pipe_coarse)
        save_images_as_grid(images, f"{output_dir}/srds_iteration_{srds_iter}.png")

        # Check convergence (line 13-14 of Algorithm 1)
        if prev_iter_images is not None:

            l1_distance = np.average(
                np.abs(
                    np.array(prev_iter_images, dtype=np.float32)
                    - np.array(images, dtype=np.float32)
                )
            )
            status = "CONVERGED" if l1_distance < tolerance else "continuing"
            tqdm.write(
                f"SRDS Iteration {srds_iter+1}: L1={l1_distance:.6f} (tolerance={tolerance}) - {status}"
            )

            if l1_distance < tolerance:
                break

        prev_iter_images = images

    # Save the final image
    images = decode_latents_to_pil(x[-len(prompts) :], pipe_coarse)
    save_images_as_grid(images, f"{output_dir}/srds_final.png")

    # Compute the difference between the final image and the ground truth
    gt_images = decode_latents_to_pil(gt_trajectory[-len(prompts) :], pipe_coarse)
    save_images_as_grid(gt_images, f"{output_dir}/srds_ddim_gt.png")

    def save_trajectory_errors_to_csv():
        diff_tensor = torch.stack(trajectory_errors)
        df = pd.DataFrame(
            diff_tensor.cpu().numpy(),
            index=[f"Iteration_{i}" for i in range(len(trajectory_errors))],
            columns=[f"Timestep_{i}" for i in range(coarse_num_inference_steps)],
        )
        df.to_csv(f"{output_dir}/trajectory_errors_table.csv", float_format="%.6f")

    def plot_trajectory_errors():
        diff_tensor = torch.stack(trajectory_errors)
        df = pd.DataFrame(
            diff_tensor.cpu().numpy(),
            index=[f"Iteration_{i}" for i in range(len(trajectory_errors))],
            columns=[f"Timestep_{i}" for i in range(coarse_num_inference_steps)],
        )

        plt.figure(figsize=(10, 6))
        timesteps = list(range(coarse_num_inference_steps))

        # Plot each iteration as a line with different colors
        for i, (iteration_name, row) in enumerate(df.iterrows()):
            plt.plot(
                timesteps, row.values, marker="o", linewidth=2, label=iteration_name
            )

        plt.xlabel("Timestep")
        plt.ylabel("Difference with Ground Truth")
        plt.title("SRDS Convergence: Difference with Ground Truth Over Timesteps")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale("log")  # Using log scale since values vary greatly
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/trajectory_errors_plot.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

    save_trajectory_errors_to_csv()
    plot_trajectory_errors()

    l1_distance = np.average(
        np.abs(
            np.array(gt_images, dtype=np.float32) - np.array(images, dtype=np.float32)
        )
    )
    print(
        f"SRDS Complete: Final L1 distance from DDIM ground truth = {l1_distance:.6f}"
    )
