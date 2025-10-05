import argparse
import os
import sys
import json
import itertools
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd

import torch

# Add parent directory to path so we can import src and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sparareal import StochasticParareal
from src.srds import SRDS
from utils.logger import setup_logging, log_info


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Run parameter sweep for SRDS/SParareal algorithms")
    
    # Required arguments
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Single prompt to test"
    )
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
        "--eta",
        nargs="+",
        type=float,
        default=[0.01, 0.1, 1.0],
        help="List of eta values to test (default: 0.01 0.1 1.0)",
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
        eta_values = args.eta
    else:
        num_samples = [1]  # SRDS doesn't use num_samples
        eta_values = [1.0]  # SRDS doesn't use eta
    
    tolerance_values = args.tolerance
    
    # Generate all combinations
    combinations = []
    for cs, fs, ns, eta, tol in itertools.product(coarse_steps, fine_steps, num_samples, eta_values, tolerance_values):
        # Skip invalid combinations
        if fs % cs != 0:
            continue
            
        combinations.append({
            'coarse_steps': cs,
            'fine_steps': fs,
            'num_samples': ns,
            'eta': eta,
            'tolerance': tol,
            'guidance_scale': args.guidance_scale,
            'height': args.height,
            'width': args.width,
            'seed': args.seed,
            'model': args.model,
        })
    
    # Limit number of experiments if specified
    if args.max_experiments and len(combinations) > args.max_experiments:
        # Sample evenly across the parameter space
        step = len(combinations) // args.max_experiments
        combinations = combinations[::step][:args.max_experiments]
    
    return combinations


def run_single_experiment(
    params: Dict[str, Any], 
    prompt: str, 
    algorithm: str,
    base_output_dir: str,
    experiment_id: int
) -> Dict[str, Any]:
    """Run a single experiment with given parameters"""
    
    # Create experiment-specific output directory
    exp_name = f"exp_{experiment_id:03d}_cs{params['coarse_steps']}-fs{params['fine_steps']}"
    if algorithm == "sparareal":
        exp_name += f"_ns{params['num_samples']}_eta{params['eta']}"
    exp_name += f"_tol{params['tolerance']}"
    
    exp_output_dir = os.path.join(base_output_dir, exp_name)
    os.makedirs(exp_output_dir, exist_ok=True)
    
    # Setup logging for this experiment
    log_file_path = os.path.join(exp_output_dir, "log.txt")
    logger = setup_logging(log_file_path)
    
    try:
        log_info(f"Starting experiment {experiment_id}")
        log_info(f"Parameters: {params}")
        
        # Set seed
        set_seed(params['seed'])
        generator = torch.Generator("cuda").manual_seed(params['seed'])
        
        # Run algorithm
        if algorithm == "srds":
            algo = SRDS(model_id=params['model'])
            images = algo(
                prompts=[prompt],
                coarse_num_inference_steps=params['coarse_steps'],
                fine_num_inference_steps=params['fine_steps'],
                tolerance=params['tolerance'],
                guidance_scale=params['guidance_scale'],
                height=params['height'],
                width=params['width'],
                generator=generator,
                output_dir=exp_output_dir,
            )
        else:  # sparareal
            algo = StochasticParareal(model_id=params['model'])
            images = algo(
                prompts=[prompt],
                coarse_num_inference_steps=params['coarse_steps'],
                fine_num_inference_steps=params['fine_steps'],
                num_samples=params['num_samples'],
                tolerance=params['tolerance'],
                guidance_scale=params['guidance_scale'],
                eta=params['eta'],
                height=params['height'],
                width=params['width'],
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
            'experiment_id': experiment_id,
            'parameters': params,
            'status': 'failed',
            'error': str(e),
            'output_dir': exp_output_dir
        }


def load_experiment_results(exp_output_dir: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Load results from a completed experiment"""
    
    results = {
        'parameters': params,
        'output_dir': exp_output_dir,
        'status': 'completed'
    }
    
    # Try to load trajectory errors
    trajectory_file = os.path.join(exp_output_dir, "trajectory_errors_table.csv")
    if os.path.exists(trajectory_file):
        try:
            df = pd.read_csv(trajectory_file)
            results['trajectory_errors'] = df.to_dict('records')
            results['final_error'] = df.iloc[-1]['Final_Image_Error'] if 'Final_Image_Error' in df.columns else None
        except Exception as e:
            print(f"Warning: Could not load trajectory errors: {e}")
    
    # Try to load log file for additional info
    log_file = os.path.join(exp_output_dir, "log.txt")
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()
                # Extract key information from log
                if "Final L1 distance from DDIM ground truth" in log_content:
                    for line in log_content.split('\n'):
                        if "Final L1 distance from DDIM ground truth" in line:
                            try:
                                l1_distance = float(line.split('=')[-1].strip())
                                results['l1_distance'] = l1_distance
                            except:
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
        if result['status'] == 'completed':
            row = result['parameters'].copy()
            row['status'] = result['status']
            row['output_dir'] = result['output_dir']
            if 'final_error' in result:
                row['final_error'] = result['final_error']
            if 'l1_distance' in result:
                row['l1_distance'] = result['l1_distance']
            summary_data.append(row)
        else:
            row = result['parameters'].copy()
            row['status'] = result['status']
            row['error'] = result.get('error', 'Unknown error')
            summary_data.append(row)
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        
        # Save CSV summary
        csv_file = os.path.join(output_dir, "sweep_summary.csv")
        df.to_csv(csv_file, index=False)
        print(f"Sweep summary saved to: {csv_file}")
        
        # Save JSON summary
        json_file = os.path.join(output_dir, "sweep_summary.json")
        with open(json_file, 'w') as f:
            json.dump({
                'prompt': prompt,
                'total_experiments': len(results),
                'completed_experiments': len([r for r in results if r['status'] == 'completed']),
                'failed_experiments': len([r for r in results if r['status'] == 'failed']),
                'results': results
            }, f, indent=2)
        print(f"Detailed results saved to: {json_file}")
        
        # Print best results
        if 'l1_distance' in df.columns:
            best_result = df.loc[df['l1_distance'].idxmin()]
            print(f"\nBest result (lowest L1 distance: {best_result['l1_distance']:.6f}):")
            print(f"  Coarse steps: {best_result['coarse_steps']}")
            print(f"  Fine steps: {best_result['fine_steps']}")
            if 'num_samples' in best_result:
                print(f"  Num samples: {best_result['num_samples']}")
            if 'eta' in best_result:
                print(f"  Eta: {best_result['eta']}")
            print(f"  Tolerance: {best_result['tolerance']}")
            print(f"  Output dir: {best_result['output_dir']}")


def main():
    args = parse_args()
    
    # Create main output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_output_dir = os.path.join(args.output_dir, f"sweep_{timestamp}_{args.algorithm}")
    os.makedirs(sweep_output_dir, exist_ok=True)
    
    # Setup main logging
    main_log_file = os.path.join(sweep_output_dir, "sweep_log.txt")
    logger = setup_logging(main_log_file)
    
    log_info("Parameter Sweep Started")
    log_info(f"Algorithm: {args.algorithm}")
    log_info(f"Prompt: {args.prompt}")
    log_info(f"Output directory: {sweep_output_dir}")
    
    # Generate parameter combinations
    combinations = generate_parameter_combinations(args)
    log_info(f"Generated {len(combinations)} parameter combinations")
    
    # Save sweep configuration
    config_file = os.path.join(sweep_output_dir, "sweep_config.json")
    with open(config_file, 'w') as f:
        json.dump({
            'prompt': args.prompt,
            'algorithm': args.algorithm,
            'parameter_lists': {
                'coarse_steps': args.coarse_steps,
                'fine_steps': args.fine_steps,
                'num_samples': args.num_samples,
                'eta': args.eta,
                'tolerance': args.tolerance,
            },
            'fixed_parameters': {
                'guidance_scale': args.guidance_scale,
                'height': args.height,
                'width': args.width,
                'seed': args.seed,
                'model': args.model,
            },
            'total_combinations': len(combinations)
        }, f, indent=2)
    
    # Run experiments
    results = []
    for i, params in enumerate(combinations):
        print(f"\nRunning experiment {i+1}/{len(combinations)}")
        print(f"Parameters: {params}")
        
        result = run_single_experiment(params, args.prompt, args.algorithm, sweep_output_dir, i)
        results.append(result)
        
        # Save intermediate results
        intermediate_file = os.path.join(sweep_output_dir, "intermediate_results.json")
        with open(intermediate_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Save final summary
    save_sweep_summary(results, sweep_output_dir, args.prompt)
    
    log_info("Parameter sweep completed!")
    print(f"\nSweep completed! Results saved to: {sweep_output_dir}")


if __name__ == "__main__":
    main()
