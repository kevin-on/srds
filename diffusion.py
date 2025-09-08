import os
from typing import List, Optional, Union

import torch
from diffusers import SchedulerMixin, UNet2DConditionModel

from utils import decode_latents_to_pil


@torch.no_grad()
def diffusion_step(
    latents: torch.Tensor,
    timestep: Union[torch.Tensor, float, int],
    prompt_embeds: torch.Tensor,
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    guidance_scale: float = 7.5,
    do_classifier_free_guidance: bool = True,
):
    """
    Single diffusion step for denoising.

    This function is extracted from the StableDiffusionPipeline's denoising loop
    to allow for custom diffusion sampling algorithms like SRDS.
    """
    # Only single timestep is supported for this function
    if isinstance(timestep, torch.Tensor) and timestep.numel() > 1:
        raise ValueError("Only single timestep is supported")

    # Validate timestep is in scheduler's timestep list
    if timestep.item() not in scheduler.timesteps:
        raise ValueError(f"Invalid timestep {timestep} not in scheduler timesteps")

    # move the timestep to the same device as the latents
    if isinstance(timestep, torch.Tensor):
        timestep = timestep.to(latents.device)

    # expand the latents if we are doing classifier free guidance
    latent_model_input = (
        torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    )

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
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

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
    Parallel diffusion step for processing multiple timesteps simultaneously.

    Note: There may be slight numerical differences compared to sequential processing
    due to batching effects in the UNet model.
    """
    # FIXME: Batched inference produces different results than sequential processing.
    # This function should produce identical outputs to diffusion_step_sequential
    # but currently does not. Root cause needs investigation and fix.

    # Validate input
    if not latents.shape[0] == timesteps.shape[0]:
        raise ValueError(f"Latents and timesteps must have the same batch size")

    timesteps = timesteps.to(latents.device)

    latent_model_input = latents.clone()

    # scheduler.scale_model_input only supports single timestep
    for i in range(0, latent_model_input.shape[0], num_prompts):
        latent_model_input[i : i + num_prompts] = scheduler.scale_model_input(
            latent_model_input[i : i + num_prompts], timesteps[i]
        )

    # expand the latents if we are doing classifier free guidance
    latent_model_input = (
        torch.cat([latent_model_input] * 2)
        if do_classifier_free_guidance
        else latent_model_input
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
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

    # TODO: Consider parallelizing this. Default scheduler.step function only supports single timestep.
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
    Sequential diffusion step for processing multiple timesteps sequentially.
    This implementation works correctly and produces expected results.
    """
    # Validate input
    if not latents.shape[0] == timesteps.shape[0]:
        raise ValueError(f"Latents and timesteps must have the same batch size")

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


@torch.no_grad()
def sanity_check_diffusion_step(
    prompts: List[str],
    num_inference_steps: int = 10,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
    seed: int = 42,
    model_id: str = "stabilityai/stable-diffusion-2",
    output_dir: Optional[str] = None,
):
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    generator = torch.Generator("cuda").manual_seed(seed)

    scheduler = DDIMScheduler.from_pretrained(
        model_id, subfolder="scheduler", timestep_spacing="trailing"
    )
    scheduler.set_timesteps(num_inference_steps)

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
    )
    pipe = pipe.to("cuda")

    # Encode prompt
    do_classifier_free_guidance = guidance_scale > 1.0
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompts,
        device=pipe.device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
    )

    # For classifier free guidance, concatenate the unconditional and text embeddings
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # Prepare latent variables
    num_channels_latents = pipe.unet.config.in_channels
    initial_latents = pipe.prepare_latents(  # line 1 of Algorithm 1
        len(prompts),
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        pipe.device,
        generator,
    )

    # Run stable diffusion pipeline to get ground truth
    output_gt = pipe(
        prompts,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        latents=initial_latents,
        generator=generator,
    )

    # Run diffusion pipeline with diffusion_step function
    latents = initial_latents.clone()
    for i in range(num_inference_steps):
        t = scheduler.timesteps[i]
        latents = diffusion_step(
            latents,
            t,
            prompt_embeds,
            pipe.unet,
            scheduler,
            guidance_scale,
            do_classifier_free_guidance,
        )

    images = decode_latents_to_pil(latents, pipe)

    # Check if the two images are the same
    if images[0].tobytes() == output_gt.images[0].tobytes():
        print(
            "✓ PASS: Custom diffusion step produces identical output to standard pipeline"
        )
    else:
        print(
            "✗ FAIL: Custom diffusion step output differs from standard pipeline - implementation may have bugs"
        )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        images[0].save(f"{output_dir}/sanity_check_result.png")
        output_gt.images[0].save(f"{output_dir}/sanity_check_gt.png")


if __name__ == "__main__":
    sanity_check_diffusion_step(
        prompts=["a beautiful painting of a cat"],
        num_inference_steps=10,
        guidance_scale=7.5,
        height=512,
        width=512,
        seed=42,
        output_dir="output/sanity_check",
    )
