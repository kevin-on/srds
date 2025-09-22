from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
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


class SRDS:
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
    ):
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype

        # Initialize the pipeline (will be loaded when needed)
        self.pipe = None

    def _load_pipeline(self) -> None:
        """Load the pipeline if not already loaded"""
        if self.pipe is None:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
            )
            self.pipe = self.pipe.to(self.device)

    @torch.no_grad()
    def __call__(
        self,
        prompts: List[str],
        coarse_num_inference_steps: int,
        fine_num_inference_steps: int,
        tolerance: float,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        generator: Optional[torch.Generator] = None,
        output_dir: str = "output",
    ) -> List[Image.Image]:
        # Load pipeline if needed
        self._load_pipeline()

        if fine_num_inference_steps % coarse_num_inference_steps != 0:
            raise ValueError(
                "The fine num inference steps must be a multiple of the coarse num inference steps"
            )

        scheduler_coarse = DDIMScheduler.from_pretrained(
            self.model_id, subfolder="scheduler", timestep_spacing="trailing"
        )
        scheduler_fine = DDIMScheduler.from_pretrained(
            self.model_id, subfolder="scheduler", timestep_spacing="trailing"
        )
        scheduler_coarse.set_timesteps(coarse_num_inference_steps)
        scheduler_fine.set_timesteps(fine_num_inference_steps)

        coarse_timesteps = scheduler_coarse.timesteps
        fine_timesteps = scheduler_fine.timesteps
        if not all(t in fine_timesteps for t in coarse_timesteps):
            raise ValueError(
                "The coarse timesteps are not a subset of the fine timesteps"
            )

        # Timestep setup
        print("SRDS Algorithm Configuration:")
        print(
            f"  Coarse steps: {len(coarse_timesteps)} | timesteps: {coarse_timesteps}"
        )
        print(f"  Fine steps: {len(fine_timesteps)} | timesteps: {fine_timesteps}")

        # Create a copy of the pipeline with the coarse scheduler
        pipe_coarse = StableDiffusionPipeline(
            vae=self.pipe.vae,
            text_encoder=self.pipe.text_encoder,
            tokenizer=self.pipe.tokenizer,
            unet=self.pipe.unet,
            scheduler=scheduler_coarse,
            safety_checker=self.pipe.safety_checker,
            feature_extractor=self.pipe.feature_extractor,
        )
        pipe_coarse = pipe_coarse.to(self.device)

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
        gt_trajectory = self._compute_gt_trajectory(
            initial_latents,
            fine_num_inference_steps,
            coarse_timesteps,
            scheduler_fine,
            prompt_embeds,
            pipe_coarse.unet,
            guidance_scale,
            do_classifier_free_guidance,
        )

        #########################################################
        # SRDS diffusion
        #########################################################

        # Initialize solution trajectories: each list has (coarse_num_inference_steps + 1) elements
        prev_final_solutions: List[torch.Tensor] = [
            initial_latents.clone() for _ in range(coarse_num_inference_steps + 1)
        ]
        prev_fine_solutions: List[torch.Tensor] = [
            initial_latents.clone() for _ in range(coarse_num_inference_steps + 1)
        ]
        prev_coarse_solutions: List[torch.Tensor] = [
            initial_latents.clone() for _ in range(coarse_num_inference_steps + 1)
        ]
        cur_final_solutions: List[torch.Tensor] = [
            initial_latents.clone() for _ in range(coarse_num_inference_steps + 1)
        ]
        cur_fine_solutions: List[torch.Tensor] = [
            initial_latents.clone() for _ in range(coarse_num_inference_steps + 1)
        ]
        cur_coarse_solutions: List[torch.Tensor] = [
            initial_latents.clone() for _ in range(coarse_num_inference_steps + 1)
        ]

        # Initialize previous solutions
        for i in range(1, coarse_num_inference_steps + 1):  # line 2 of Algorithm 1
            prev_final_solutions[i] = diffusion_step(  # line 3 of Algorithm 1
                prev_final_solutions[i - 1],
                coarse_timesteps[i - 1],
                prompt_embeds,
                pipe_coarse.unet,
                scheduler_coarse,
                guidance_scale,
                do_classifier_free_guidance,
            )

        prev_coarse_solutions = [x.clone() for x in prev_final_solutions]

        self._sanity_check_initialization(
            pipe_coarse,
            prompts,
            coarse_num_inference_steps,
            guidance_scale,
            initial_latents,
            prev_final_solutions[-1],
            output_dir,
        )

        trajectory_errors: List[List[float]] = (
            []
        )  # [iteration][timestep] error matrix for convergence analysis
        prev_iter_images: Optional[List[Image.Image]] = (
            None  # Previous final image for L1 convergence check
        )
        for srds_iter in tqdm(
            range(coarse_num_inference_steps), desc="SRDS Iterations"
        ):  # line 6 of Algorithm 1

            # cur_fine_solutions starts from prev_final_solutions
            for i in range(1, coarse_num_inference_steps + 1):
                cur_fine_solutions[i] = prev_final_solutions[i - 1].clone()

            tqdm.write(
                f"SRDS Iteration {srds_iter+1}/{coarse_num_inference_steps} - Processing fine steps"
            )
            for i in range(1, coarse_num_inference_steps + 1):  # line 7 of Algorithm 1
                timestep_start = coarse_timesteps[i - 1]
                timestep_end = (
                    coarse_timesteps[i] if i < coarse_num_inference_steps else -1
                )
                cur_fine_solutions[i] = diffusion_step(
                    latents=cur_fine_solutions[i],
                    timestep=timestep_start,
                    timestep_end=timestep_end,
                    prompt_embeds=prompt_embeds,
                    unet=pipe_coarse.unet,
                    scheduler=scheduler_fine,
                    guidance_scale=guidance_scale,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                )

            tqdm.write(
                f"SRDS Iteration {srds_iter+1}/{coarse_num_inference_steps} - Processing coarse sweep"
            )
            for i in range(1, coarse_num_inference_steps + 1):  # line 9 of Algorithm 1
                cur_coarse_solutions[i] = diffusion_step(  # line 10 of Algorithm 1
                    cur_final_solutions[i - 1],
                    coarse_timesteps[i - 1],
                    prompt_embeds,
                    pipe_coarse.unet,
                    scheduler_coarse,
                    guidance_scale,
                    do_classifier_free_guidance,
                )
                cur_final_solutions[i] = cur_fine_solutions[i] + (
                    cur_coarse_solutions[i] - prev_coarse_solutions[i]
                )  # line 11 of Algorithm 1

            diff_in_cur_iter = []
            for i in range(1, coarse_num_inference_steps + 1):
                diff_in_cur_iter.append(
                    torch.norm(cur_final_solutions[i] - gt_trajectory[i]).item()
                )
            trajectory_errors.append(diff_in_cur_iter)

            # Save final images of every iteration
            images = decode_latents_to_pil(cur_final_solutions[-1], pipe_coarse)
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

            # Update previous solutions (line 12 of Algorithm 1)
            prev_coarse_solutions[:] = cur_coarse_solutions
            prev_fine_solutions[:] = cur_fine_solutions
            prev_final_solutions[:] = cur_final_solutions
            prev_iter_images = images

        # Save outputs and analyze results
        images = decode_latents_to_pil(cur_final_solutions[-1], pipe_coarse)
        gt_images = decode_latents_to_pil(gt_trajectory[-1], pipe_coarse)

        self._save_outputs(
            images, gt_images, trajectory_errors, coarse_num_inference_steps, output_dir
        )

        l1_distance = np.average(
            np.abs(
                np.array(gt_images, dtype=np.float32)
                - np.array(images, dtype=np.float32)
            )
        )
        print(
            f"SRDS Complete: Final L1 distance from DDIM ground truth = {l1_distance:.6f}"
        )

        return images

    def _compute_gt_trajectory(
        self,
        initial_latents: torch.Tensor,
        fine_num_inference_steps: int,
        coarse_timesteps: torch.Tensor,
        scheduler_fine: DDIMScheduler,
        prompt_embeds: torch.Tensor,
        unet: torch.nn.Module,
        guidance_scale: float,
        do_classifier_free_guidance: bool,
    ) -> List[torch.Tensor]:
        """
        Compute ground truth DDIM trajectory using fine scheduler.

        Returns:
            List[torch.Tensor]: Ground truth latents at each coarse timestep, with length = coarse_num_inference_steps + 1.
        """
        gt_trajectory = []
        latents_gt = initial_latents.clone()
        for i in range(fine_num_inference_steps):
            t = scheduler_fine.timesteps[i]
            if t in coarse_timesteps:
                gt_trajectory.append(latents_gt.clone())
            latents_gt = diffusion_step(
                latents_gt,
                t,
                prompt_embeds,
                unet,
                scheduler_fine,
                guidance_scale,
                do_classifier_free_guidance,
            )
        gt_trajectory.append(latents_gt.clone())
        return gt_trajectory

    def _sanity_check_initialization(
        self,
        pipe_coarse: StableDiffusionPipeline,
        prompts: List[str],
        coarse_num_inference_steps: int,
        guidance_scale: float,
        initial_latents: torch.Tensor,
        final_latents: torch.Tensor,
        output_dir: str,
    ) -> None:
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

        initialized_images = decode_latents_to_pil(final_latents, pipe_coarse)
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

    def _save_outputs(
        self,
        images: List[Image.Image],
        gt_images: List[Image.Image],
        trajectory_errors: List[List[float]],
        coarse_num_inference_steps: int,
        output_dir: str,
    ) -> None:
        """Save final outputs and analysis"""
        # Save the final image
        save_images_as_grid(images, f"{output_dir}/srds_final.png")

        # Save ground truth
        save_images_as_grid(gt_images, f"{output_dir}/srds_ddim_gt.png")

        # Save trajectory errors to CSV
        if trajectory_errors:
            df = pd.DataFrame(
                np.array(trajectory_errors),
                index=[f"Iteration_{i}" for i in range(len(trajectory_errors))],
                columns=[f"Timestep_{i}" for i in range(coarse_num_inference_steps)],
            )
            df.to_csv(f"{output_dir}/trajectory_errors_table.csv", float_format="%.6f")

            # Plot trajectory errors
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
