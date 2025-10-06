#!/usr/bin/env python3
"""
Unit tests to verify that ddim_step_with_eta produces the same results as
scheduler.step() with eta parameter.
"""

import os
import sys
import unittest

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

# Add parent directory to path to import diffusion module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.diffusion import _single_diffusion_step, ddim_step_with_eta


class TestDiffusionEta(unittest.TestCase):
    """Test suite for ddim_step_with_eta function equivalence with scheduler.step(eta)."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are expensive to create."""
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.model_id = "stabilityai/stable-diffusion-2"

        # Initialize scheduler with known configuration
        cls.scheduler = DDIMScheduler.from_pretrained(
            cls.model_id, subfolder="scheduler", timestep_spacing="trailing"
        )
        cls.scheduler.set_timesteps(20)

        # Initialize minimal pipeline components for testing
        cls.pipe = StableDiffusionPipeline.from_pretrained(cls.model_id)
        cls.pipe = cls.pipe.to(cls.device)

    def setUp(self):
        """Set up test data for each test."""
        # Create reproducible test data
        self.generator = torch.Generator(device=self.device).manual_seed(42)

        # Create test latents
        self.latents = torch.randn(
            1,
            4,
            64,
            64,
            device=self.device,
            generator=self.generator,
            dtype=torch.float32,
        )

        # Create test prompt embeddings
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt=["test prompt"],
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
        self.prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # Test timestep (middle of the schedule)
        self.timestep = self.scheduler.timesteps[10]

    def test_ddim_step_with_eta_zero(self):
        """Test that ddim_step_with_eta with eta=0 matches deterministic DDIM."""
        print("\n=== Test: ddim_step_with_eta with eta=0 ===")

        # Get deterministic DDIM result using standard diffusion step
        x_t_minus_1_deterministic = _single_diffusion_step(
            self.latents.clone(),
            self.timestep,
            self.prompt_embeds,
            self.pipe.unet,
            self.scheduler,
            guidance_scale=7.5,
            do_classifier_free_guidance=True,
        )

        # Apply ddim_step_with_eta with eta=0 (should return the same as deterministic)
        result_eta_zero = ddim_step_with_eta(
            self.latents.clone(),
            x_t_minus_1_deterministic,
            self.timestep,
            self.scheduler,
            eta=0.0,
            generator=self.generator,
        )

        # They should be identical for eta=0
        max_diff = torch.abs(result_eta_zero - x_t_minus_1_deterministic).max().item()
        print(f"Max difference with eta=0: {max_diff:.2e}")

        self.assertLess(
            max_diff,
            1e-6,
            "ddim_step_with_eta with eta=0 should return deterministic result",
        )
        print("✓ PASS: eta=0 returns deterministic result")

    def test_ddim_step_with_eta_vs_scheduler_with_eta(self):
        """Test ddim_step_with_eta against scheduler.step() with eta parameter."""
        print("\n=== Test: ddim_step_with_eta vs scheduler.step(eta) ===")

        eta_values = [0.0, 0.1, 0.5, 1.0]

        for eta in eta_values:
            with self.subTest(eta=eta):
                print(f"\n--- Testing eta={eta} ---")

                # Create a fresh scheduler instance for each test to avoid state issues
                test_scheduler = DDIMScheduler.from_pretrained(
                    self.model_id, subfolder="scheduler", timestep_spacing="trailing"
                )
                test_scheduler.set_timesteps(20)

                # Get noise prediction using UNet
                generator_clone = torch.Generator(device=self.device).manual_seed(42)
                latent_model_input = torch.cat([self.latents] * 2)
                latent_model_input = test_scheduler.scale_model_input(
                    latent_model_input, self.timestep
                )

                noise_pred = self.pipe.unet(
                    latent_model_input,
                    self.timestep,
                    encoder_hidden_states=self.prompt_embeds,
                    return_dict=False,
                )[0]

                # Apply classifier-free guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

                # Method 1: Use scheduler.step() with eta parameter
                if hasattr(test_scheduler, "_get_variance"):
                    # For newer versions of diffusers that support eta in step()
                    try:
                        scheduler_result = test_scheduler.step(
                            noise_pred,
                            self.timestep,
                            self.latents.clone(),
                            eta=eta,
                            generator=generator_clone,
                            return_dict=False,
                        )[0]

                        # Get deterministic result for ddim_step_with_eta
                        x_t_minus_1_deterministic = test_scheduler.step(
                            noise_pred,
                            self.timestep,
                            self.latents.clone(),
                            eta=0.0,
                            return_dict=False,
                        )[0]

                        # Method 2: Use our custom ddim_step_with_eta
                        generator_clone2 = torch.Generator(device=self.device).manual_seed(42)
                        custom_result = ddim_step_with_eta(
                            self.latents.clone(),
                            x_t_minus_1_deterministic,
                            self.timestep,
                            test_scheduler,
                            eta=eta,
                            generator=generator_clone2,
                        )

                        # Compare results
                        max_diff = torch.abs(scheduler_result - custom_result).max().item()
                        mean_diff = torch.abs(scheduler_result - custom_result).mean().item()

                        print(f"Max difference: {max_diff:.6f}")
                        print(f"Mean difference: {mean_diff:.6f}")

                        # Allow for some numerical differences due to different implementations
                        tolerance = 1e-3 if eta > 0 else 1e-6
                        self.assertLess(
                            max_diff,
                            tolerance,
                            f"ddim_step_with_eta differs too much from "
                            f"scheduler.step() for eta={eta}",
                        )
                        print(f"✓ PASS: eta={eta} results match within tolerance {tolerance}")

                    except TypeError:
                        # Fallback: scheduler.step() doesn't support eta parameter
                        print("⚠ SKIP: scheduler.step() doesn't support eta parameter")
                        self._test_manual_eta_implementation(eta, test_scheduler, noise_pred)

                else:
                    print("⚠ SKIP: scheduler doesn't have _get_variance method")
                    self._test_manual_eta_implementation(eta, test_scheduler, noise_pred)

    def _test_manual_eta_implementation(self, eta, scheduler, noise_pred):
        """Test our implementation against manual DDIM+eta calculations."""
        print(f"Testing manual implementation for eta={eta}")

        # Get deterministic result first
        x_t_minus_1_deterministic = scheduler.step(
            noise_pred, self.timestep, self.latents.clone(), return_dict=False
        )[0]

        # Apply our custom function
        generator_test = torch.Generator(device=self.device).manual_seed(42)
        custom_result = ddim_step_with_eta(
            self.latents.clone(),
            x_t_minus_1_deterministic,
            self.timestep,
            scheduler,
            eta=eta,
            generator=generator_test,
        )

        # For eta=0, should match deterministic
        if eta == 0.0:
            max_diff = torch.abs(custom_result - x_t_minus_1_deterministic).max().item()
            self.assertLess(max_diff, 1e-6, "eta=0 should match deterministic result")
            print(f"✓ PASS: eta=0 matches deterministic (diff: {max_diff:.2e})")
        else:
            # For eta>0, should be different from deterministic (stochastic)
            max_diff = torch.abs(custom_result - x_t_minus_1_deterministic).max().item()
            self.assertGreater(max_diff, 1e-6, f"eta={eta} should be different from deterministic")
            print(f"✓ PASS: eta={eta} is stochastic (diff: {max_diff:.6f})")

    def test_eta_parameter_validation(self):
        """Test that eta parameter validation works correctly."""
        print("\n=== Test: eta parameter validation ===")

        # Get a deterministic result for testing
        x_t_minus_1 = _single_diffusion_step(
            self.latents.clone(),
            self.timestep,
            self.prompt_embeds,
            self.pipe.unet,
            self.scheduler,
            guidance_scale=7.5,
            do_classifier_free_guidance=True,
        )

        # Test invalid eta values
        invalid_etas = [-0.1, 1.1, -1.0, 2.0]

        for invalid_eta in invalid_etas:
            with self.subTest(eta=invalid_eta):
                with self.assertRaises(ValueError):
                    ddim_step_with_eta(
                        self.latents.clone(),
                        x_t_minus_1,
                        self.timestep,
                        self.scheduler,
                        eta=invalid_eta,
                    )

        print("✓ PASS: Invalid eta values correctly raise ValueError")

    def test_reproducibility_with_generator(self):
        """Test that results are reproducible when using the same generator seed."""
        print("\n=== Test: Reproducibility with generator ===")

        # Get deterministic result
        x_t_minus_1 = _single_diffusion_step(
            self.latents.clone(),
            self.timestep,
            self.prompt_embeds,
            self.pipe.unet,
            self.scheduler,
            guidance_scale=7.5,
            do_classifier_free_guidance=True,
        )

        eta = 0.7
        seed = 12345

        # Run twice with same seed
        gen1 = torch.Generator(device=self.device).manual_seed(seed)
        result1 = ddim_step_with_eta(
            self.latents.clone(),
            x_t_minus_1,
            self.timestep,
            self.scheduler,
            eta=eta,
            generator=gen1,
        )

        gen2 = torch.Generator(device=self.device).manual_seed(seed)
        result2 = ddim_step_with_eta(
            self.latents.clone(),
            x_t_minus_1,
            self.timestep,
            self.scheduler,
            eta=eta,
            generator=gen2,
        )

        # Should be identical
        max_diff = torch.abs(result1 - result2).max().item()
        self.assertLess(max_diff, 1e-8, "Results should be identical with same generator seed")

        # Run with different seed - should be different
        gen3 = torch.Generator(device=self.device).manual_seed(seed + 1)
        result3 = ddim_step_with_eta(
            self.latents.clone(),
            x_t_minus_1,
            self.timestep,
            self.scheduler,
            eta=eta,
            generator=gen3,
        )

        max_diff_different = torch.abs(result1 - result3).max().item()
        self.assertGreater(max_diff_different, 1e-6, "Results should differ with different seeds")

        print(f"✓ PASS: Same seed gives identical results (diff: {max_diff:.2e})")
        print(f"✓ PASS: Different seed gives different results (diff: {max_diff_different:.6f})")


def demo_eta_effects():
    """Demonstrate the effects of different eta values on sampling."""
    print("\n" + "=" * 60)
    print("DEMO: Effects of different eta values")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "stabilityai/stable-diffusion-2"

    # Initialize components
    scheduler = DDIMScheduler.from_pretrained(
        model_id, subfolder="scheduler", timestep_spacing="trailing"
    )
    scheduler.set_timesteps(10)  # Fewer steps for demo

    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to(device)

    # Create test data
    generator = torch.Generator(device).manual_seed(42)
    latents = torch.randn(1, 4, 64, 64, device=device, generator=generator, dtype=torch.float32)

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=["a beautiful sunset"],
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # Test different eta values
    eta_values = [0.0, 0.2, 0.5, 0.8, 1.0]
    timestep = scheduler.timesteps[5]  # Middle timestep

    # Get deterministic baseline
    x_t_minus_1_deterministic = _single_diffusion_step(
        latents.clone(),
        timestep,
        prompt_embeds,
        pipe.unet,
        scheduler,
        guidance_scale=7.5,
        do_classifier_free_guidance=True,
    )

    print(f"\nTesting timestep: {timestep}")
    print(f"Input shape: {latents.shape}")
    print(f"Deterministic output mean: {x_t_minus_1_deterministic.mean().item():.6f}")
    print(f"Deterministic output std: {x_t_minus_1_deterministic.std().item():.6f}")

    print("\nEta value effects:")
    for eta in eta_values:
        # Use fixed generator for reproducible results
        gen = torch.Generator(device).manual_seed(12345)

        result = ddim_step_with_eta(
            latents.clone(),
            x_t_minus_1_deterministic,
            timestep,
            scheduler,
            eta=eta,
            generator=gen,
        )

        # Compare with deterministic
        diff_from_deterministic = torch.abs(result - x_t_minus_1_deterministic).mean().item()
        result_std = result.std().item()

        print(f"  eta={eta:.1f}: mean_diff={diff_from_deterministic:.6f}, std={result_std:.6f}")

    print("\n✓ Demo completed - shows increasing stochasticity with higher eta values")


def run_tests():
    """Run all tests and print summary."""
    print("=" * 60)
    print("Testing ddim_step_with_eta vs scheduler.step(eta) equivalence")
    print("=" * 60)

    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestDiffusionEta)

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED!")
        print("ddim_step_with_eta function works correctly")
    else:
        print("❌ SOME TESTS FAILED!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    print("=" * 60)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    if success:
        # Run demo if tests pass
        demo_eta_effects()
    else:
        exit(1)
