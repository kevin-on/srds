from typing import Optional, Union

import torch
from diffusers import SchedulerMixin, UNet2DConditionModel


@torch.no_grad()
def ddim_step_with_eta(
    x_t: torch.Tensor,
    x_t_minus_1: torch.Tensor,
    timestep: Union[torch.Tensor, float, int],
    scheduler: SchedulerMixin,
    eta: float = 0.0,
    generator: Optional[torch.Generator] = None,
):
    """
    Compute DDIM step with eta > 0 given x_t and x_{t-1} (from eta=0 DDIM).

    This function mimics the official DDIMScheduler.step() with eta parameter.
    For eta=0: deterministic DDIM (x_{t-1} is given)
    For eta>0: stochastic interpolation between DDIM and DDPM

    Args:
        x_t: Current latent at timestep t
        x_t_minus_1: Target latent at timestep t-1 (from eta=0 DDIM)
        timestep: Current timestep t
        scheduler: DDIM scheduler
        eta: DDIM eta parameter (0=deterministic, 1=full stochastic)
        generator: Random generator for reproducible sampling

    Returns:
        New x_{t-1} sample with stochasticity controlled by eta
    """
    if eta < 0 or eta > 1:
        raise ValueError(f"eta must be between 0 and 1, got {eta}")

    if eta == 0.0:
        # Pure deterministic case - return the given x_{t-1}
        return x_t_minus_1

    # Get scheduler parameters
    timestep_val = timestep.item() if isinstance(timestep, torch.Tensor) else timestep
    if timestep_val not in scheduler.timesteps:
        raise ValueError(f"Invalid timestep {timestep_val} not in scheduler timesteps")

    # Get step index and alpha values from scheduler
    step_index = (scheduler.timesteps == timestep_val).nonzero().item()

    # Get alpha values at current and previous timesteps
    alpha_prod_t = scheduler.alphas_cumprod[timestep_val]

    if step_index == len(scheduler.timesteps) - 1:
        alpha_prod_t_prev = torch.tensor(1.0, device=x_t.device, dtype=x_t.dtype)
    else:
        prev_timestep = scheduler.timesteps[step_index + 1]
        alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep]

    # Extract noise prediction from the deterministic DDIM result
    # From DDIM equations:
    # x_t = sqrt(α_t) * x_0 + sqrt(1-α_t) * ε
    # x_{t-1} = sqrt(α_{t-1}) * x_0 + sqrt(1-α_{t-1}) * ε (for eta=0)

    sqrt_alpha_prod_t = alpha_prod_t**0.5
    sqrt_one_minus_alpha_prod_t = (1 - alpha_prod_t) ** 0.5
    sqrt_alpha_prod_t_prev = alpha_prod_t_prev**0.5
    sqrt_one_minus_alpha_prod_t_prev = (1 - alpha_prod_t_prev) ** 0.5

    denominator = (
        sqrt_alpha_prod_t_prev * sqrt_one_minus_alpha_prod_t
        - sqrt_alpha_prod_t * sqrt_one_minus_alpha_prod_t_prev
    )

    if abs(denominator) < 1e-8:
        # Fallback for numerical stability
        pred_x0 = (
            x_t_minus_1
            - sqrt_one_minus_alpha_prod_t_prev
            * (x_t - sqrt_alpha_prod_t * x_t_minus_1 / sqrt_alpha_prod_t_prev)
            / sqrt_one_minus_alpha_prod_t
        ) / sqrt_alpha_prod_t_prev
    else:
        # More stable solution
        pred_x0 = (
            x_t_minus_1 * sqrt_one_minus_alpha_prod_t - x_t * sqrt_one_minus_alpha_prod_t_prev
        ) / denominator

    # Extract noise prediction
    noise_pred = (x_t - sqrt_alpha_prod_t * pred_x0) / sqrt_one_minus_alpha_prod_t

    # Compute variance following the DDIM paper
    # variance = σ_t^2 = η^2 * β_{t-1} * (1 - α_t / α_{t-1}) / (1 - α_t)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    # Standard DDIM variance calculation
    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    std_dev_t = eta * (variance**0.5)

    # Generate random noise
    if generator is not None:
        noise = torch.randn(x_t.shape, generator=generator, device=x_t.device, dtype=x_t.dtype)
    else:
        noise = torch.randn_like(x_t)

    # Compute the new sample with stochasticity
    # Following DDIM paper: x_{t-1} = sqrt(α_{t-1}) * x_0 + sqrt(1-α_{t-1}-σ_t^2) * ε + σ_t * z
    # where z is random noise

    # Direction pointing to x_t (for DDIM deterministic part)
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** 0.5 * noise_pred

    # Combine deterministic and stochastic parts
    prev_sample = sqrt_alpha_prod_t_prev * pred_x0 + pred_sample_direction + std_dev_t * noise

    return prev_sample


@torch.no_grad()
def diffusion_step(
    latents: torch.Tensor,
    timestep: Union[torch.Tensor, float, int],
    prompt_embeds: torch.Tensor,
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    guidance_scale: float = 7.5,
    do_classifier_free_guidance: bool = True,
    timestep_end: Optional[Union[torch.Tensor, float, int]] = None,
):
    """
    Single or multiple diffusion steps for denoising.

    This function is extracted from the StableDiffusionPipeline's denoising loop
    to allow for custom diffusion sampling algorithms like SRDS.

    Args:
        timestep_end: Optional end timestep. When provided, multiple diffusion steps
                     will be executed from timestep up to but not including
                     timestep_end (exclusive). Use timestep_end=-1 to process
                     until the very last timestep.
    """
    # Convert timesteps to int for easier comparison
    timestep_val = timestep.item() if isinstance(timestep, torch.Tensor) else timestep

    if timestep_end is not None:
        # Find the range of timesteps to process
        scheduler_timesteps = scheduler.timesteps.tolist()

        # Get start index
        try:
            start_idx = scheduler_timesteps.index(timestep_val)
        except ValueError as e:
            raise ValueError(
                f"Start timestep {timestep_val} not found in scheduler timesteps"
            ) from e

        # Handle different timestep_end cases
        if timestep_end == -1:
            # Special case: process until the very end (last timestep in scheduler)
            end_idx = len(scheduler_timesteps)
        else:
            timestep_end_val = (
                timestep_end.item() if isinstance(timestep_end, torch.Tensor) else timestep_end
            )
            try:
                end_idx = scheduler_timesteps.index(timestep_end_val)
            except ValueError as e:
                raise ValueError(
                    f"End timestep {timestep_end_val} not found in scheduler timesteps"
                ) from e

        # Ensure we're going in the right direction (start_idx < end_idx for denoising)
        if start_idx >= end_idx:
            raise ValueError(
                f"Invalid timestep range: start index {start_idx} "
                f"must be less than end index {end_idx}"
            )

        # Run multiple diffusion steps (exclusive of end_idx)
        current_latents = latents
        for i in range(start_idx, end_idx):
            current_timestep = scheduler_timesteps[i]
            current_latents = _single_diffusion_step(
                current_latents,
                current_timestep,
                prompt_embeds,
                unet,
                scheduler,
                guidance_scale,
                do_classifier_free_guidance,
            )

        return current_latents
    else:
        # Single timestep processing (original behavior)
        return _single_diffusion_step(
            latents,
            timestep,
            prompt_embeds,
            unet,
            scheduler,
            guidance_scale,
            do_classifier_free_guidance,
        )


def _single_diffusion_step(
    latents: torch.Tensor,
    timestep: Union[torch.Tensor, float, int],
    prompt_embeds: torch.Tensor,
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    guidance_scale: float = 7.5,
    do_classifier_free_guidance: bool = True,
):
    """
    Internal function for a single diffusion step.
    """
    # Only single timestep is supported for this function
    if isinstance(timestep, torch.Tensor) and timestep.numel() > 1:
        raise ValueError("Only single timestep is supported")

    # Validate timestep is in scheduler's timestep list
    timestep_val = timestep.item() if isinstance(timestep, torch.Tensor) else timestep
    if timestep_val not in scheduler.timesteps:
        raise ValueError(f"Invalid timestep {timestep_val} not in scheduler timesteps")

    # move the timestep to the same device as the latents
    if isinstance(timestep, torch.Tensor):
        timestep = timestep.to(latents.device)

    # expand the latents if we are doing classifier free guidance
    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

    # predict the noise residual
    noise_pred = unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=prompt_embeds,
        return_dict=False,
    )[0]

    # perform guidance
    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    latents = scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
    return latents


@torch.no_grad()
def diffusion_step_batched(
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    num_prompts: int,
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    guidance_scale: float = 7.5,
    do_classifier_free_guidance: bool = True,
):
    """
    DEPRECATED: Use diffusion_step with timestep_end parameter instead.

    Parallel diffusion step for processing multiple timesteps simultaneously.

    Note: There may be slight numerical differences compared to sequential processing
    due to batching effects in the UNet model.
    """
    # FIXME: Batched inference produces different results than sequential processing.
    # This function should produce identical outputs to diffusion_step_sequential
    # but currently does not. Root cause needs investigation and fix.

    # Validate input
    if not latents.shape[0] == timesteps.shape[0]:
        raise ValueError("Latents and timesteps must have the same batch size")

    timesteps = timesteps.to(latents.device)

    latent_model_input = latents.clone()

    # scheduler.scale_model_input only supports single timestep
    for i in range(0, latent_model_input.shape[0], num_prompts):
        latent_model_input[i : i + num_prompts] = scheduler.scale_model_input(
            latent_model_input[i : i + num_prompts], timesteps[i]
        )

    # expand the latents if we are doing classifier free guidance
    latent_model_input = (
        torch.cat([latent_model_input] * 2) if do_classifier_free_guidance else latent_model_input
    )

    # expand the timesteps
    timesteps = torch.cat([timesteps] * 2) if do_classifier_free_guidance else timesteps

    # Broadcast prompt_embeds to have same batch size as latent_model_input
    repeat_factor = latent_model_input.shape[0] // prompt_embeds.shape[0]
    if do_classifier_free_guidance:
        negative_prompt_embeds, prompt_embeds = prompt_embeds.chunk(2)
        prompt_embeds = torch.cat(
            [negative_prompt_embeds] * repeat_factor + [prompt_embeds] * repeat_factor
        )
    else:
        prompt_embeds = torch.cat([prompt_embeds] * repeat_factor)

    # predict the noise residual
    noise_pred = unet(
        latent_model_input,
        timesteps,
        encoder_hidden_states=prompt_embeds,
        return_dict=False,
    )[0]

    # perform guidance
    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # TODO: Consider parallelizing this. Default scheduler.step function only
    # supports single timestep.
    for i in range(latents.shape[0]):
        latents[i : i + 1] = scheduler.step(
            noise_pred[i : i + 1],
            timesteps[i],
            latents[i : i + 1],
            return_dict=False,
        )[0]

    return latents


@torch.no_grad()
def diffusion_step_sequential(
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    num_prompts: int,
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    guidance_scale: float = 7.5,
    do_classifier_free_guidance: bool = True,
):
    """
    DEPRECATED: Use diffusion_step with timestep_end parameter instead.

    Sequential diffusion step for processing multiple timesteps sequentially.
    This implementation works correctly and produces expected results.
    """
    # Validate input
    if not latents.shape[0] == timesteps.shape[0]:
        raise ValueError("Latents and timesteps must have the same batch size")

    timesteps = timesteps.to(latents.device)

    for i in range(0, latents.shape[0], num_prompts):
        latents[i : i + num_prompts] = diffusion_step(
            latents[i : i + num_prompts],
            timesteps[i],
            prompt_embeds,
            unet,
            scheduler,
            guidance_scale,
            do_classifier_free_guidance,
        )

    return latents
