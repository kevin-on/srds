#!/usr/bin/env python3
"""
Test to verify that SRDS and SParareal produce exactly the same error trajectory
when num_sample=1 in SParareal.

This test ensures that when SParareal uses only one sample, it reduces to
the standard SRDS algorithm and produces identical results.
"""

import os
import tempfile
from typing import List, Tuple

import numpy as np
import torch

from sparareal import StochasticParareal
from srds import SRDS


def setup_test_environment() -> Tuple[str, dict]:
    """Setup test environment with consistent parameters"""
    # Create temporary directory for outputs
    temp_dir = tempfile.mkdtemp(prefix="test_srds_sparareal_")

    # Common test parameters
    test_params = {
        "prompts": ["a cat sitting on a table"],
        "coarse_num_inference_steps": 5,
        "fine_num_inference_steps": 10,
        "tolerance": 0.1,
        "guidance_scale": 7.5,
        "height": 512,
        "width": 512,
    }

    return temp_dir, test_params


def extract_trajectory_errors(output_dir: str) -> List[List[float]]:
    """Extract trajectory errors from CSV file"""
    import pandas as pd

    csv_path = os.path.join(output_dir, "trajectory_errors_table.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Trajectory errors CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, index_col=0)
    return df.values.tolist()


def compare_trajectories(
    srds_errors: List[List[float]],
    sparareal_errors: List[List[float]],
    tolerance: float = 1e-6,
) -> Tuple[bool, List[Tuple[int, int, float]]]:
    """
    Compare two trajectory error lists and return whether they match within tolerance.

    Returns:
        Tuple of (all_match, differences) where differences is a list of
        (iteration, timestep, difference) for any mismatches.
    """
    if len(srds_errors) != len(sparareal_errors):
        return False, [(-1, -1, abs(len(srds_errors) - len(sparareal_errors)))]

    differences = []
    all_match = True

    for i, (srds_iter, sparareal_iter) in enumerate(zip(srds_errors, sparareal_errors)):
        if len(srds_iter) != len(sparareal_iter):
            differences.append((i, -1, abs(len(srds_iter) - len(sparareal_iter))))
            all_match = False
            continue

        for j, (srds_val, sparareal_val) in enumerate(zip(srds_iter, sparareal_iter)):
            diff = abs(srds_val - sparareal_val)
            if diff > tolerance:
                differences.append((i, j, diff))
                all_match = False

    return all_match, differences


def test_srds_sparareal_equivalence():
    """Main test function"""
    print("=" * 80)
    print("Testing SRDS and SParareal Equivalence (num_samples=1)")
    print("=" * 80)

    # Setup test environment
    temp_dir, test_params = setup_test_environment()

    try:
        # Set up consistent random seed for reproducibility
        torch.manual_seed(42)
        generator = torch.Generator().manual_seed(42)

        print(f"Test parameters:")
        for key, value in test_params.items():
            print(f"  {key}: {value}")
        print(f"Output directory: {temp_dir}")
        print()

        # Create output directories
        srds_output_dir = os.path.join(temp_dir, "srds")
        sparareal_output_dir = os.path.join(temp_dir, "sparareal")
        os.makedirs(srds_output_dir, exist_ok=True)
        os.makedirs(sparareal_output_dir, exist_ok=True)

        # Initialize both algorithms
        print("Initializing SRDS...")
        srds = SRDS(
            model_id="stabilityai/stable-diffusion-2",
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        print("Initializing SParareal...")
        sparareal = StochasticParareal(
            model_id="stabilityai/stable-diffusion-2",
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        # Run SRDS
        print("\n" + "-" * 40)
        print("Running SRDS...")
        print("-" * 40)

        # Reset seed for SRDS
        torch.manual_seed(42)
        generator_srds = torch.Generator().manual_seed(42)

        srds_images = srds(
            **test_params,
            generator=generator_srds,
            output_dir=srds_output_dir,
        )

        # Run SParareal with num_samples=1
        print("\n" + "-" * 40)
        print("Running SParareal (num_samples=1)...")
        print("-" * 40)

        # Reset seed for SParareal to ensure same initial conditions
        torch.manual_seed(42)
        generator_sparareal = torch.Generator().manual_seed(42)

        sparareal_images = sparareal(
            **test_params,
            num_samples=1,  # This should make SParareal equivalent to SRDS
            generator=generator_sparareal,
            output_dir=sparareal_output_dir,
        )

        # Extract and compare trajectory errors
        print("\n" + "-" * 40)
        print("Comparing Results...")
        print("-" * 40)

        srds_errors = extract_trajectory_errors(srds_output_dir)
        sparareal_errors = extract_trajectory_errors(sparareal_output_dir)

        print(
            f"SRDS trajectory shape: {len(srds_errors)} iterations x {len(srds_errors[0]) if srds_errors else 0} timesteps"
        )
        print(
            f"SParareal trajectory shape: {len(sparareal_errors)} iterations x {len(sparareal_errors[0]) if sparareal_errors else 0} timesteps"
        )

        # Compare trajectories
        tolerance = 1e-6
        trajectories_match, differences = compare_trajectories(
            srds_errors, sparareal_errors, tolerance
        )

        # Print detailed comparison
        print(f"\nTrajectory Error Comparison (tolerance: {tolerance}):")

        if trajectories_match:
            print("✅ PASS: Trajectory errors match within tolerance!")
            print(
                "   SRDS and SParareal produce identical error trajectories when num_samples=1"
            )
        else:
            print("❌ FAIL: Trajectory errors do not match!")
            print(f"   Found {len(differences)} differences:")

            for i, (iter_idx, timestep_idx, diff) in enumerate(
                differences[:10]
            ):  # Show first 10 differences
                if iter_idx == -1:
                    print(
                        f"     {i+1}. Different number of iterations: difference = {diff}"
                    )
                elif timestep_idx == -1:
                    print(
                        f"     {i+1}. Iteration {iter_idx}: Different number of timesteps: difference = {diff}"
                    )
                else:
                    print(
                        f"     {i+1}. Iteration {iter_idx}, Timestep {timestep_idx}: difference = {diff:.2e}"
                    )

            if len(differences) > 10:
                print(f"     ... and {len(differences) - 10} more differences")

        # Compare final images numerically
        print(f"\nFinal Image Comparison:")
        if len(srds_images) == len(sparareal_images):
            total_pixel_diff = 0
            for i, (srds_img, sparareal_img) in enumerate(
                zip(srds_images, sparareal_images)
            ):
                srds_array = np.array(srds_img, dtype=np.float32)
                sparareal_array = np.array(sparareal_img, dtype=np.float32)
                pixel_diff = np.mean(np.abs(srds_array - sparareal_array))
                total_pixel_diff += pixel_diff
                print(f"  Image {i}: Average pixel difference = {pixel_diff:.6f}")

            avg_pixel_diff = total_pixel_diff / len(srds_images)
            print(f"  Overall average pixel difference: {avg_pixel_diff:.6f}")

            if (
                avg_pixel_diff < 1.0
            ):  # Very small difference expected due to floating point precision
                print("✅ PASS: Final images are nearly identical!")
            else:
                print("❌ FAIL: Final images differ significantly!")
        else:
            print(
                f"❌ FAIL: Different number of images: SRDS={len(srds_images)}, SParareal={len(sparareal_images)}"
            )

        # Print summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)

        if trajectories_match:
            print("✅ OVERALL RESULT: PASS")
            print("   SRDS and SParareal produce equivalent results when num_samples=1")
            print(
                "   This confirms that SParareal correctly reduces to SRDS in the single-sample case."
            )
        else:
            print("❌ OVERALL RESULT: FAIL")
            print("   SRDS and SParareal produce different results when num_samples=1")
            print("   This indicates a potential bug in the SParareal implementation.")

        print(f"\nOutput files saved to: {temp_dir}")
        print("   - SRDS results: srds/")
        print("   - SParareal results: sparareal/")

        return trajectories_match

    except Exception as e:
        print(f"❌ TEST FAILED with exception: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Note: Not cleaning up temp_dir so user can inspect results
        pass


if __name__ == "__main__":
    # Ensure we're in the right directory and have required dependencies
    try:
        import diffusers
        import einops
        import matplotlib
        import pandas
        import torch
        import tqdm
        import transformers
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        exit(1)

    # Run the test
    success = test_srds_sparareal_equivalence()
    exit(0 if success else 1)
