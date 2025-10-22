from typing import List, Optional, Union

import numpy as np
import torch
from diffusers import DDIMScheduler, SchedulerMixin, StableDiffusionPipeline
from PIL import Image
from tqdm import tqdm

from src.diffusion import ddim_step_with_eta, diffusion_step
from src.srds import SRDS
from utils.logger import get_logger, get_tqdm_logger
from utils.utils import decode_latents_to_pil, parse_sample_type, save_images_as_grid


class StochasticParareal(SRDS):
    @torch.no_grad()
    def _sample_latents_ddim(
        self,
        num_samples: int,
        x_t: torch.Tensor,
        x_t_minus_1: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        scheduler: SchedulerMixin,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> List[torch.Tensor]:
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")

        return [
            x_t_minus_1,
            *[
                ddim_step_with_eta(
                    x_t,
                    x_t_minus_1,
                    timestep,
                    scheduler,
                    eta,
                    generator,
                )
                for _ in range(num_samples - 1)
            ],
        ]

    def _sample_latents_dir(
        self,
        num_samples: int,
        latent: torch.Tensor,
        direction: torch.Tensor,
        scale: float,
    ) -> List[torch.Tensor]:
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")

        if num_samples == 1:
            return [latent]
        # Generate correction factors around 0 with the given scale
        factors = np.linspace(-scale, scale, num_samples)
        if 0.0 in factors:
            factors = [0.0] + [f for f in factors if f != 0.0]

        return [latent + direction * factor for factor in factors]

    @torch.no_grad()
    def __call__(
        self,
        prompts: List[str],
        coarse_num_inference_steps: int,
        fine_num_inference_steps: int,
        num_samples: int,
        tolerance: float,
        guidance_scale: float = 7.5,
        sample_type: str = "ddim,eta=1.0",  # "ddim,eta=" / "dir,shoot="
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
            raise ValueError("The coarse timesteps are not a subset of the fine timesteps")

        # Timestep setup
        get_logger().info("SParareal Algorithm Configuration:")
        get_logger().info(
            f"  Coarse steps: {len(coarse_timesteps)} | timesteps: {coarse_timesteps}"
        )
        get_logger().info(f"  Fine steps: {len(fine_timesteps)} | timesteps: {fine_timesteps}")
        get_logger().info(f"  Num samples: {num_samples}")
        get_logger().info(f"  Sample type: {sample_type}")

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
        # Stochastic Parareal diffusion
        #########################################################

        class TrajectorySegment:
            def __init__(self, start: torch.Tensor, end: torch.Tensor):
                self.start = start
                self.end = end

        # Initialize solution trajectories: each list has (coarse_num_inference_steps + 1) elements
        prev_fine_prediction: List[TrajectorySegment] = [
            TrajectorySegment(initial_latents.clone(), initial_latents.clone())
            for _ in range(coarse_num_inference_steps + 1)
        ]  # F_n(U_{k-1}^{n-1})
        prev_coarse_prediction: List[torch.Tensor] = [
            initial_latents.clone() for _ in range(coarse_num_inference_steps + 1)
        ]  # G_n(U_{k-1}^{n-1})
        prev_corrected_solution: List[torch.Tensor] = [
            initial_latents.clone() for _ in range(coarse_num_inference_steps + 1)
        ]  # U_k^n
        cur_fine_prediction: List[TrajectorySegment] = [
            TrajectorySegment(initial_latents.clone(), initial_latents.clone())
            for _ in range(coarse_num_inference_steps + 1)
        ]  # F_n(U_k^{n-1})
        cur_coarse_prediction: List[torch.Tensor] = [
            initial_latents.clone() for _ in range(coarse_num_inference_steps + 1)
        ]  # G_n(U_k^{n-1})
        cur_corrected_solution: List[torch.Tensor] = [
            initial_latents.clone() for _ in range(coarse_num_inference_steps + 1)
        ]  # U_{k+1}^n

        # Sparareal-specific variables
        fine_trajectory_samples: List[List[TrajectorySegment]] = [
            [] for _ in range(coarse_num_inference_steps + 1)
        ]
        coarse_prediction_from_optimal: List[torch.Tensor] = [
            initial_latents.clone() for _ in range(coarse_num_inference_steps + 1)
        ]  # G_n(alpha_k^{n-1})

        # Initialize previous solutions
        for i in range(1, coarse_num_inference_steps + 1):  # line 2 of Algorithm 1
            prev_coarse_prediction[i] = diffusion_step(  # line 3 of Algorithm 1
                prev_coarse_prediction[i - 1],
                coarse_timesteps[i - 1],
                prompt_embeds,
                pipe_coarse.unet,
                scheduler_coarse,
                guidance_scale,
                do_classifier_free_guidance,
            )

        # line 4 of Algorithm 1
        prev_corrected_solution = [x.clone() for x in prev_coarse_prediction]

        self._sanity_check_initialization(
            pipe_coarse,
            prompts,
            coarse_num_inference_steps,
            guidance_scale,
            initial_latents,
            prev_corrected_solution[-1],
            output_dir,
        )

        gt_trajectory_errors: List[
            List[float]
        ] = []  # [iteration][timestep] error matrix for ground truth comparison
        trajectory_convergences: List[
            List[float]
        ] = []  # [iteration][timestep] error matrix for convergence analysis
        prev_images: Optional[List[Image.Image]] = (
            None  # Previous final image for L1 convergence check
        )
        for srds_iter in tqdm(range(coarse_num_inference_steps), desc="SRDS Iterations"):
            # line 6 of Algorithm 1
            # cur_fine_prediction starts from prev_corrected_solution
            sample_config = parse_sample_type(sample_type)
            if sample_config["method"] == "ddim":
                for i in range(1, coarse_num_inference_steps + 1):
                    # FIXME
                    if i == 1:
                        fine_trajectory_samples[i] = [
                            TrajectorySegment(x, x)
                            for x in [prev_corrected_solution[i - 1].clone()]
                        ]
                    else:
                        fine_trajectory_samples[i] = [
                            TrajectorySegment(x, x)
                            for x in self._sample_latents_ddim(
                                num_samples=num_samples,
                                x_t=prev_corrected_solution[i - 2].clone(),
                                x_t_minus_1=prev_corrected_solution[i - 1].clone(),
                                timestep=coarse_timesteps[i - 2],
                                scheduler=scheduler_coarse,
                                generator=None,
                                **sample_config["kwargs"],
                            )
                        ]
            elif sample_config["method"] == "dir":
                for i in range(1, coarse_num_inference_steps + 1):
                    fine_trajectory_samples[i] = [
                        TrajectorySegment(x, x)
                        for x in self._sample_latents_dir(
                            num_samples=num_samples,
                            latent=prev_corrected_solution[i - 1].clone(),
                            direction=(
                                prev_fine_prediction[i - 1].end.clone()
                                - coarse_prediction_from_optimal[i - 1].clone()
                            ),
                            **sample_config["kwargs"],
                        )
                    ]
            get_tqdm_logger().write(
                f"SParareal Iteration {srds_iter + 1}/{coarse_num_inference_steps} "
                "- Processing fine steps"
            )
            for i in range(1, coarse_num_inference_steps + 1):  # line 7 of Algorithm 1
                timestep_start = coarse_timesteps[i - 1]
                timestep_end = coarse_timesteps[i] if i < coarse_num_inference_steps else -1
                for j in range(len(fine_trajectory_samples[i])):
                    fine_trajectory_samples[i][j].end = diffusion_step(
                        latents=fine_trajectory_samples[i][j].start,
                        timestep=timestep_start,
                        timestep_end=timestep_end,
                        prompt_embeds=prompt_embeds,
                        unet=pipe_coarse.unet,
                        scheduler=scheduler_fine,
                        guidance_scale=guidance_scale,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                    )

            # Select optimal trajectory
            for i in range(1, coarse_num_inference_steps + 1):
                distances = [
                    torch.norm(x.start - cur_fine_prediction[i - 1].end).item()
                    for x in fine_trajectory_samples[i]
                ]
                min_idx = distances.index(min(distances))
                cur_fine_prediction[i] = fine_trajectory_samples[i][min_idx]

                get_tqdm_logger().write(
                    f"  Timestep {i}: Selected candidate {min_idx} (is_original: {min_idx == 0})"
                )
                get_tqdm_logger().write(f"    Distances: {[f'{d:.6f}' for d in distances]}")

            # Compute coarse prediction from the start point of each optimal trajectory
            for i in range(1, coarse_num_inference_steps + 1):
                coarse_prediction_from_optimal[i] = diffusion_step(
                    cur_fine_prediction[i].start,
                    coarse_timesteps[i - 1],
                    prompt_embeds,
                    pipe_coarse.unet,
                    scheduler_coarse,
                    guidance_scale,
                    do_classifier_free_guidance,
                )

            get_tqdm_logger().write(
                f"SParareal Iteration {srds_iter + 1}/{coarse_num_inference_steps} "
                "- Processing coarse sweep"
            )
            for i in range(1, coarse_num_inference_steps + 1):  # line 9 of Algorithm 1
                cur_coarse_prediction[i] = diffusion_step(  # line 10 of Algorithm 1
                    cur_corrected_solution[i - 1],
                    coarse_timesteps[i - 1],
                    prompt_embeds,
                    pipe_coarse.unet,
                    scheduler_coarse,
                    guidance_scale,
                    do_classifier_free_guidance,
                )
                cur_corrected_solution[i] = cur_fine_prediction[i].end + (
                    cur_coarse_prediction[i] - coarse_prediction_from_optimal[i]
                )  # line 11 of Algorithm 1

            errors = []
            for i in range(1, coarse_num_inference_steps + 1):
                errors.append(torch.norm(cur_corrected_solution[i] - gt_trajectory[i]).item())
            gt_trajectory_errors.append(errors)

            convergences = []
            for i in range(1, coarse_num_inference_steps + 1):
                convergences.append(
                    torch.norm(cur_corrected_solution[i] - prev_corrected_solution[i]).item()
                )
            trajectory_convergences.append(convergences)

            # Save final images of every iteration
            images = decode_latents_to_pil(cur_corrected_solution[-1], pipe_coarse)
            save_images_as_grid(images, f"{output_dir}/srds_iteration_{srds_iter}.png")

            # Check convergence (line 13-14 of Algorithm 1)
            if prev_images is not None:
                l1_distance = np.average(
                    np.abs(
                        np.array(prev_images, dtype=np.float32) - np.array(images, dtype=np.float32)
                    )
                )
                status = "CONVERGED" if l1_distance < tolerance else "continuing"
                get_tqdm_logger().write(
                    f"SParareal Iteration {srds_iter + 1}: L1={l1_distance:.6f} "
                    f"(tolerance={tolerance}) - {status}"
                )

                if l1_distance < tolerance:
                    break

            # Update previous solutions (line 12 of Algorithm 1)
            prev_coarse_prediction[:] = cur_coarse_prediction
            prev_fine_prediction[:] = cur_fine_prediction
            prev_corrected_solution[:] = cur_corrected_solution
            prev_images = images

        # Save outputs and analyze results
        images = decode_latents_to_pil(cur_corrected_solution[-1], pipe_coarse)
        gt_images = decode_latents_to_pil(gt_trajectory[-1], pipe_coarse)

        self._save_outputs(
            images,
            gt_images,
            gt_trajectory_errors,
            trajectory_convergences,
            coarse_num_inference_steps,
            output_dir,
        )

        l1_distance = np.average(
            np.abs(np.array(gt_images, dtype=np.float32) - np.array(images, dtype=np.float32))
        )
        get_logger().info(
            f"SParareal Complete: Final L1 distance from DDIM ground truth = {l1_distance:.6f}"
        )

        return images
