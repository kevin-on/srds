import argparse
import itertools
import json
import os
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import torch

from src.sparareal import StochasticParareal
from src.srds import SRDS
from utils.logger import log_info, setup_logging
from utils.metrics import get_convergence_metrics, get_segment_selection_metrics


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run parameter sweep for SRDS/SParareal algorithms"
    )

    # Required arguments
    parser.add_argument("--prompt", type=str, required=True, help="Single prompt to test")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        required=True,
        help="Output directory for sweep results",
    )

    # Algorithm selection
    parser.add_argument(
        "--algorithm",
        "-a",
        type=str,
        choices=["srds", "sparareal"],
        default="sparareal",
        help="Algorithm to use: srds or sparareal",
    )

    # Parameter ranges for sweep
    parser.add_argument(
        "--coarse-steps",
        nargs="+",
        type=int,
        default=[5, 10, 15],
        help="List of coarse steps to test (default: 5 10 15)",
    )
    parser.add_argument(
        "--fine-steps",
        nargs="+",
        type=int,
        default=[50, 100, 150],
        help="List of fine steps to test (default: 50 100 150)",
    )
    parser.add_argument(
        "--num-samples",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        help="List of num_samples to test (default: 1 5 10)",
    )
    parser.add_argument(
        "--sample-type",
        nargs="+",
        type=str,
        default=["ddim,eta=0.01", "ddim,eta=0.1", "ddim,eta=1.0"],
        help="List of sample types to test (default: ddim,eta=0.01 ddim,eta=0.1 ddim,eta=1.0)",
    )
    parser.add_argument(
        "--tolerance",
        nargs="+",
        type=float,
        default=[0.01, 0.05, 0.1],
        help="List of tolerance values to test (default: 0.01 0.05 0.1)",
    )

    # Fixed parameters
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

    # Sweep control
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=None,
        help="Maximum number of experiments to run (default: all combinations)",
    )

    return parser.parse_args()


def generate_parameter_combinations(args) -> List[Dict[str, Any]]:
    """Generate all parameter combinations for the sweep"""

    # Generate parameter ranges
    coarse_steps = args.coarse_steps
    fine_steps = args.fine_steps

    if args.algorithm == "sparareal":
        num_samples = args.num_samples
        sample_types = args.sample_type
    else:
        num_samples = [1]  # SRDS doesn't use num_samples
        sample_types = ["ddim,eta=1.0"]  # SRDS doesn't use sample_type

    tolerance_values = args.tolerance

    # Generate all combinations
    combinations = []
    for cs, fs, ns, sample_type, tol in itertools.product(
        coarse_steps, fine_steps, num_samples, sample_types, tolerance_values
    ):
        # Skip invalid combinations
        if fs % cs != 0:
            continue

        combinations.append(
            {
                "coarse_steps": cs,
                "fine_steps": fs,
                "num_samples": ns,
                "sample_type": sample_type,
                "tolerance": tol,
                "guidance_scale": args.guidance_scale,
                "height": args.height,
                "width": args.width,
                "seed": args.seed,
                "model": args.model,
            }
        )

    # Limit number of experiments if specified
    if args.max_experiments and len(combinations) > args.max_experiments:
        # Sample evenly across the parameter space
        step = len(combinations) // args.max_experiments
        combinations = combinations[::step][: args.max_experiments]

    return combinations


def run_single_experiment(
    params: Dict[str, Any],
    prompt: str,
    algorithm: str,
    base_output_dir: str,
    experiment_id: int,
) -> Dict[str, Any]:
    """Run a single experiment with given parameters"""

    # Create experiment-specific output directory
    exp_name = f"exp_{experiment_id:03d}_cs{params['coarse_steps']}-fs{params['fine_steps']}"
    if algorithm == "sparareal":
        # Clean sample_type for directory name (replace commas and equals with underscores)
        clean_sample_type = params["sample_type"].replace(",", "_").replace("=", "_")
        exp_name += f"_ns{params['num_samples']}_{clean_sample_type}"
    exp_name += f"_tol{params['tolerance']}"

    exp_output_dir = os.path.join(base_output_dir, exp_name)
    os.makedirs(exp_output_dir, exist_ok=True)

    # Setup logging for this experiment
    log_file_path = os.path.join(exp_output_dir, "log.txt")
    setup_logging(log_file_path)

    try:
        log_info(f"Starting experiment {experiment_id}")
        log_info(f"Parameters: {params}")

        # Set seed
        set_seed(params["seed"])
        generator = torch.Generator("cuda").manual_seed(params["seed"])

        # Run algorithm
        if algorithm == "srds":
            algo = SRDS(model_id=params["model"])
            algo(
                prompts=[prompt],
                coarse_num_inference_steps=params["coarse_steps"],
                fine_num_inference_steps=params["fine_steps"],
                tolerance=params["tolerance"],
                guidance_scale=params["guidance_scale"],
                height=params["height"],
                width=params["width"],
                generator=generator,
                output_dir=exp_output_dir,
            )
        else:  # sparareal
            algo = StochasticParareal(model_id=params["model"])
            algo(
                prompts=[prompt],
                coarse_num_inference_steps=params["coarse_steps"],
                fine_num_inference_steps=params["fine_steps"],
                num_samples=params["num_samples"],
                tolerance=params["tolerance"],
                guidance_scale=params["guidance_scale"],
                sample_type=params["sample_type"],
                height=params["height"],
                width=params["width"],
                generator=generator,
                output_dir=exp_output_dir,
            )

        # Load results
        results = load_experiment_results(exp_output_dir, params)
        log_info(f"Experiment {experiment_id} completed successfully")

        return results

    except Exception as e:
        log_info(f"Experiment {experiment_id} failed: {str(e)}")
        return {
            "experiment_id": experiment_id,
            "parameters": params,
            "status": "failed",
            "error": str(e),
            "output_dir": exp_output_dir,
        }


def load_experiment_results(exp_output_dir: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Load results from a completed experiment"""

    results = {
        "parameters": params,
        "output_dir": exp_output_dir,
        "status": "completed",
    }

    # Load convergence metrics
    convergence_metrics = get_convergence_metrics(exp_output_dir)
    results.update(convergence_metrics)

    # Load segment selection metrics for sparareal
    if params.get("algorithm") == "sparareal" or "sparareal" in exp_output_dir:
        segment_metrics = get_segment_selection_metrics(exp_output_dir)
        if segment_metrics:
            results["alternative_selection_rate"] = segment_metrics["alternative_selection_rate"]
            results["original_selection_rate"] = segment_metrics["original_selection_rate"]

    # Try to load L1 distance from log file
    log_file = os.path.join(exp_output_dir, "log.txt")
    if os.path.exists(log_file):
        try:
            with open(log_file) as f:
                log_content = f.read()

                # Extract L1 distance information
                if "Final L1 distance from DDIM ground truth" in log_content:
                    for line in log_content.split("\n"):
                        if "Final L1 distance from DDIM ground truth" in line:
                            try:
                                l1_distance = float(line.split("=")[-1].strip())
                                results["l1_distance"] = l1_distance
                            except (ValueError, IndexError):
                                pass
                            break
        except Exception as e:
            print(f"Warning: Could not parse log file: {e}")

    return results


def save_sweep_summary(results: List[Dict[str, Any]], output_dir: str, prompt: str):
    """Save summary of all sweep results"""

    # Convert to DataFrame for analysis
    summary_data = []
    for result in results:
        if result["status"] == "completed":
            row = result["parameters"].copy()
            row["status"] = result["status"]
            row["output_dir"] = result["output_dir"]
            if "final_error" in result:
                row["final_error"] = result["final_error"]
            if "l1_distance" in result:
                row["l1_distance"] = result["l1_distance"]
            summary_data.append(row)
        else:
            row = result["parameters"].copy()
            row["status"] = result["status"]
            row["error"] = result.get("error", "Unknown error")
            summary_data.append(row)

    if summary_data:
        df = pd.DataFrame(summary_data)

        # Save CSV summary
        csv_file = os.path.join(output_dir, "sweep_summary.csv")
        df.to_csv(csv_file, index=False)
        print(f"Sweep summary saved to: {csv_file}")

        # Save JSON summary
        json_file = os.path.join(output_dir, "sweep_summary.json")
        with open(json_file, "w") as f:
            json.dump(
                {
                    "prompt": prompt,
                    "total_experiments": len(results),
                    "completed_experiments": len(
                        [r for r in results if r["status"] == "completed"]
                    ),
                    "failed_experiments": len([r for r in results if r["status"] == "failed"]),
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"Detailed results saved to: {json_file}")

        # Print key results summary
        print("\n=== SWEEP SUMMARY ===")

        # Show results table
        print("\nResults Summary:")
        key_cols = [
            "coarse_steps",
            "fine_steps",
            "num_samples",
            "sample_type",
            "tolerance",
        ]
        if "total_iterations" in df.columns:
            key_cols.append("total_iterations")
        if "alternative_selection_rate" in df.columns:
            key_cols.append("alternative_selection_rate")

        available_cols = [col for col in key_cols if col in df.columns]
        print(df[available_cols].to_string(index=False))

        # Best results
        if "total_iterations" in df.columns and df["total_iterations"].notna().any():
            best_convergence = df.loc[df["total_iterations"].idxmin()]
            print(f"\n⚡ Fastest Convergence: {best_convergence['total_iterations']} iterations")
            print(
                f"  Settings: cs={best_convergence['coarse_steps']}, "
                f"fs={best_convergence['fine_steps']}, "
                f"ns={best_convergence.get('num_samples', 'N/A')}, "
                f"sample_type={best_convergence.get('sample_type', 'N/A')}"
            )

        if (
            "alternative_selection_rate" in df.columns
            and df["alternative_selection_rate"].notna().any()
        ):
            avg_alt_rate = df["alternative_selection_rate"].mean()
            print(f"\n🎯 Average Alternative Selection Rate: {avg_alt_rate:.1%}")

            most_exploratory = df.loc[df["alternative_selection_rate"].idxmax()]
            print(
                f"  Most exploratory: {most_exploratory['alternative_selection_rate']:.1%} "
                f"(cs={most_exploratory['coarse_steps']}, "
                f"ns={most_exploratory.get('num_samples', 'N/A')}, "
                f"sample_type={most_exploratory.get('sample_type', 'N/A')})"
            )


def main():
    args = parse_args()

    # Create main output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_output_dir = os.path.join(args.output_dir, f"sweep_{timestamp}_{args.algorithm}")
    os.makedirs(sweep_output_dir, exist_ok=True)

    # Setup main logging
    main_log_file = os.path.join(sweep_output_dir, "sweep_log.txt")
    setup_logging(main_log_file)

    log_info("Parameter Sweep Started")
    log_info(f"Algorithm: {args.algorithm}")
    log_info(f"Prompt: {args.prompt}")
    log_info(f"Output directory: {sweep_output_dir}")

    # Generate parameter combinations
    combinations = generate_parameter_combinations(args)
    log_info(f"Generated {len(combinations)} parameter combinations")

    # Save sweep configuration
    config_file = os.path.join(sweep_output_dir, "sweep_config.json")
    with open(config_file, "w") as f:
        json.dump(
            {
                "prompt": args.prompt,
                "algorithm": args.algorithm,
                "parameter_lists": {
                    "coarse_steps": args.coarse_steps,
                    "fine_steps": args.fine_steps,
                    "num_samples": args.num_samples,
                    "sample_type": args.sample_type,
                    "tolerance": args.tolerance,
                },
                "fixed_parameters": {
                    "guidance_scale": args.guidance_scale,
                    "height": args.height,
                    "width": args.width,
                    "seed": args.seed,
                    "model": args.model,
                },
                "total_combinations": len(combinations),
            },
            f,
            indent=2,
        )

    # Run experiments
    results = []
    for i, params in enumerate(combinations):
        print(f"\nRunning experiment {i + 1}/{len(combinations)}")
        print(f"Parameters: {params}")

        result = run_single_experiment(params, args.prompt, args.algorithm, sweep_output_dir, i)
        results.append(result)

        # Save intermediate results
        intermediate_file = os.path.join(sweep_output_dir, "intermediate_results.json")
        with open(intermediate_file, "w") as f:
            json.dump(results, f, indent=2)

    # Save final summary
    save_sweep_summary(results, sweep_output_dir, args.prompt)

    log_info("Parameter sweep completed!")
    print(f"\nSweep completed! Results saved to: {sweep_output_dir}")


if __name__ == "__main__":
    main()
