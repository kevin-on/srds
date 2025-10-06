#!/usr/bin/env python3
"""
Comprehensive test suite for diffusion_step function to ensure all functionality works correctly.
"""

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

from src.diffusion import diffusion_step


def test_diffusion_step_comprehensive():
    """Test all variants of diffusion_step function."""
    print("Starting comprehensive diffusion_step tests...")

    model_id = "stabilityai/stable-diffusion-2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize pipeline components
    scheduler = DDIMScheduler.from_pretrained(
        model_id, subfolder="scheduler", timestep_spacing="trailing"
    )
    scheduler.set_timesteps(20)  # Use 20 steps for thorough testing

    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler)
    pipe = pipe.to(device)

    # Prepare test inputs
    prompt = "a beautiful landscape"
    generator = torch.Generator(device).manual_seed(42)

    # Encode prompt
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=[prompt],
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # Prepare initial latents
    latents = pipe.prepare_latents(1, 4, 512, 512, prompt_embeds.dtype, device, generator)

    print(f"Scheduler timesteps: {scheduler.timesteps.tolist()}")
    print(f"Total timesteps: {len(scheduler.timesteps)}")

    # Test 1: Single step (original behavior)
    print("\n=== Test 1: Single Step ===")
    latents_single = diffusion_step(
        latents.clone(),
        scheduler.timesteps[0],
        prompt_embeds,
        pipe.unet,
        scheduler,
        guidance_scale=7.5,
        do_classifier_free_guidance=True,
    )
    print("✓ Single step execution works")

    # Test 2: Multi-step with explicit end timestep (exclusive)
    print("\n=== Test 2: Multi-step (Exclusive End) ===")
    start_timestep = scheduler.timesteps[0]
    end_timestep = scheduler.timesteps[3]  # Will process steps 0, 1, 2 (exclusive of 3)

    latents_multi_exclusive = diffusion_step(
        latents.clone(),
        start_timestep,
        prompt_embeds,
        pipe.unet,
        scheduler,
        guidance_scale=7.5,
        do_classifier_free_guidance=True,
        timestep_end=end_timestep,
    )
    print(f"✓ Multi-step (exclusive) works: {start_timestep} -> {end_timestep} (exclusive)")

    # Test 3: Verify multi-step equals sequential single steps
    print("\n=== Test 3: Multi-step vs Sequential Equivalence ===")
    latents_sequential = latents.clone()
    for i in range(3):  # Steps 0, 1, 2 (same as exclusive range above)
        latents_sequential = diffusion_step(
            latents_sequential,
            scheduler.timesteps[i],
            prompt_embeds,
            pipe.unet,
            scheduler,
            guidance_scale=7.5,
            do_classifier_free_guidance=True,
        )

    # Check if results are identical
    diff = torch.abs(latents_multi_exclusive - latents_sequential).max().item()
    if diff < 1e-6:
        print(f"✓ Multi-step matches sequential steps (max diff: {diff:.2e})")
    else:
        print(f"✗ Multi-step differs from sequential steps (max diff: {diff:.2e})")
        return False

    # Test 4: Process to the end using timestep_end=-1
    print("\n=== Test 4: Process to End (timestep_end=-1) ===")
    latents_to_end = diffusion_step(
        latents.clone(),
        scheduler.timesteps[0],
        prompt_embeds,
        pipe.unet,
        scheduler,
        guidance_scale=7.5,
        do_classifier_free_guidance=True,
        timestep_end=-1,
    )
    print("✓ Process to end (timestep_end=-1) works")

    # Test 5: Verify process-to-end equals full sequential processing
    print("\n=== Test 5: Process to End vs Full Sequential Equivalence ===")
    latents_full_sequential = latents.clone()
    for i in range(len(scheduler.timesteps)):
        latents_full_sequential = diffusion_step(
            latents_full_sequential,
            scheduler.timesteps[i],
            prompt_embeds,
            pipe.unet,
            scheduler,
            guidance_scale=7.5,
            do_classifier_free_guidance=True,
        )

    # Check if results are identical
    diff_end = torch.abs(latents_to_end - latents_full_sequential).max().item()
    if diff_end < 1e-6:
        print(f"✓ Process-to-end matches full sequential (max diff: {diff_end:.2e})")
    else:
        print(f"✗ Process-to-end differs from full sequential (max diff: {diff_end:.2e})")
        return False

    # Test 6: Error handling - invalid timestep range
    print("\n=== Test 6: Error Handling ===")
    try:
        # Should fail: end timestep before start timestep
        diffusion_step(
            latents.clone(),
            scheduler.timesteps[5],
            prompt_embeds,
            pipe.unet,
            scheduler,
            timestep_end=scheduler.timesteps[2],  # Earlier timestep
        )
        print("✗ Error handling failed - should have raised exception for invalid range")
        return False
    except ValueError as e:
        print(f"✓ Error handling works for invalid timestep range: {str(e)}")

    # Test 7: Error handling - invalid timestep
    try:
        # Should fail: timestep not in scheduler
        diffusion_step(
            latents.clone(),
            9999,  # Invalid timestep
            prompt_embeds,
            pipe.unet,
            scheduler,
        )
        print("✗ Error handling failed - should have raised exception for invalid timestep")
        return False
    except ValueError as e:
        print(f"✓ Error handling works for invalid timestep: {str(e)}")

    # Test 8: Different step ranges
    print("\n=== Test 8: Various Step Ranges ===")

    # Small range
    diffusion_step(
        latents.clone(),
        scheduler.timesteps[10],
        prompt_embeds,
        pipe.unet,
        scheduler,
        timestep_end=scheduler.timesteps[12],
    )
    print("✓ Small range (2 steps) works")

    # Single step using timestep_end (should be equivalent to no timestep_end)
    latents_single_with_end = diffusion_step(
        latents.clone(),
        scheduler.timesteps[0],
        prompt_embeds,
        pipe.unet,
        scheduler,
        timestep_end=scheduler.timesteps[1],  # Next timestep (exclusive)
    )

    # Compare with single step without timestep_end
    diff_single = torch.abs(latents_single - latents_single_with_end).max().item()
    if diff_single < 1e-6:
        print(f"✓ Single step with timestep_end matches without (max diff: {diff_single:.2e})")
    else:
        print(f"✗ Single step with timestep_end differs (max diff: {diff_single:.2e})")
        return False

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! diffusion_step function works correctly")
    print("=" * 50)
    return True


def test_diffusion_step_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n\nTesting edge cases...")

    model_id = "stabilityai/stable-diffusion-2"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Very small scheduler for edge case testing
    scheduler = DDIMScheduler.from_pretrained(
        model_id, subfolder="scheduler", timestep_spacing="trailing"
    )
    scheduler.set_timesteps(3)  # Minimal steps

    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler)
    pipe = pipe.to(device)

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=["test"],
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    latents = pipe.prepare_latents(1, 4, 512, 512, prompt_embeds.dtype, device, None)

    print(f"Edge case scheduler timesteps: {scheduler.timesteps.tolist()}")

    # Edge case 1: Process only last step to end
    print("\n=== Edge Case 1: Last Step to End ===")
    try:
        diffusion_step(
            latents.clone(),
            scheduler.timesteps[-1],  # Last timestep
            prompt_embeds,
            pipe.unet,
            scheduler,
            timestep_end=-1,
        )
        print("✓ Last step to end works")
    except Exception as e:
        print(f"✗ Last step to end failed: {e}")
        return False

    # Edge case 2: First step to second step
    print("\n=== Edge Case 2: First to Second Step ===")
    try:
        diffusion_step(
            latents.clone(),
            scheduler.timesteps[0],
            prompt_embeds,
            pipe.unet,
            scheduler,
            timestep_end=scheduler.timesteps[1],
        )
        print("✓ First to second step works")
    except Exception as e:
        print(f"✗ First to second step failed: {e}")
        return False

    print("✓ All edge cases passed!")
    return True


@torch.no_grad()
def test_pipeline_compatibility():
    """Test that diffusion_step produces identical results to StableDiffusionPipeline."""
    print("\n\nTesting pipeline compatibility...")

    model_id = "stabilityai/stable-diffusion-2"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    import os
    import sys

    from diffusers import DDIMScheduler, StableDiffusionPipeline

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import decode_latents_to_pil

    num_inference_steps = 10
    guidance_scale = 7.5
    seed = 42
    prompts = ["a beautiful painting of a cat"]

    generator = torch.Generator(device).manual_seed(seed)

    scheduler = DDIMScheduler.from_pretrained(
        model_id, subfolder="scheduler", timestep_spacing="trailing"
    )
    scheduler.set_timesteps(num_inference_steps)

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
    )
    pipe = pipe.to(device)

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
    initial_latents = pipe.prepare_latents(
        len(prompts),
        num_channels_latents,
        512,
        512,
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
        print("✓ PASS: Custom diffusion step produces identical output to standard pipeline")
        return True
    else:
        print("✗ FAIL: Custom diffusion step output differs from standard pipeline")
        return False


if __name__ == "__main__":
    success = test_diffusion_step_comprehensive()
    if success:
        success = test_diffusion_step_edge_cases()
    if success:
        success = test_pipeline_compatibility()

    if success:
        print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
    else:
        print("\n❌ SOME TESTS FAILED!")
        exit(1)
