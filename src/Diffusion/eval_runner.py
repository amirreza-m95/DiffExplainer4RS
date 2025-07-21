#!/usr/bin/env python3
"""
eval_runner.py - Automated evaluation runner for all diffusion model checkpoints

This script automatically discovers all checkpoint files in the checkpoints/diffusionModels
folder and runs evaluation for each one using the eval_Multidim.py script, which can handle
multi-dimensional checkpoints with non-consecutive layer indices.

Usage:
    python eval_runner.py [options]

Options:
    --data_name: Dataset name (default: ML1M)
    --recommender_name: Recommender name (default: VAE)
    --batch_size: Batch size (default: 128)
    --learning_rate: Learning rate (default: 0.001)
    --lambda_cf: Lambda counterfactual (default: 0.5)
    --lambda_l1: Lambda L1 (default: 1.0)
    --lambda_preserve: Lambda preserve (default: 1.0)
    --parallel: Run evaluations in parallel (default: False)
    --max_workers: Maximum number of parallel workers (default: 4)
    --filter_pattern: Filter checkpoints by pattern (e.g., "trial" to only run trial checkpoints)
    --exclude_pattern: Exclude checkpoints by pattern
    --skip_architecture_mismatch: Skip checkpoints with architecture mismatches
    --skip_incompatible: Skip incompatible checkpoints
    --dry_run: Show what would be run without actually running (default: False)
"""

import os
import sys
import subprocess
import argparse
import time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eval_runner.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def discover_checkpoints(checkpoints_dir: Path, filter_pattern: Optional[str] = None, exclude_pattern: Optional[str] = None) -> List[Path]:
    """
    Discover all checkpoint files in the given directory.
    
    Args:
        checkpoints_dir: Path to the checkpoints directory
        filter_pattern: Optional pattern to filter checkpoints (e.g., "trial")
        exclude_pattern: Optional pattern to exclude checkpoints
    
    Returns:
        List of checkpoint file paths
    """
    if not checkpoints_dir.exists():
        logger.error(f"Checkpoints directory does not exist: {checkpoints_dir}")
        return []
    
    checkpoint_files = []
    for file_path in checkpoints_dir.glob("*.pt"):
        filename = file_path.name
        
        # Apply filter pattern
        if filter_pattern and filter_pattern not in filename:
            continue
            
        # Apply exclude pattern
        if exclude_pattern and exclude_pattern in filename:
            continue
            
        checkpoint_files.append(file_path)
    
    # Sort by filename for consistent ordering
    checkpoint_files.sort(key=lambda x: x.name)
    
    logger.info(f"Discovered {len(checkpoint_files)} checkpoint files")
    return checkpoint_files

def run_evaluation(checkpoint_path: Path, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run evaluation for a single checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        args: Evaluation arguments
    
    Returns:
        Dictionary with evaluation results
    """
    checkpoint_name = checkpoint_path.name
    logger.info(f"Starting evaluation for checkpoint: {checkpoint_name}")
    
    # Build command
    cmd = [
        sys.executable, "eval_Multidim.py",
        "--checkpoint", checkpoint_name,
        "--data_name", args["data_name"],
        "--recommender_name", args["recommender_name"],
        "--batch_size", str(args["batch_size"]),
        "--learning_rate", str(args["learning_rate"]),
        "--lambda_cf", str(args["lambda_cf"]),
        "--lambda_l1", str(args["lambda_l1"]),
        "--lambda_preserve", str(args["lambda_preserve"])
    ]
    
    # Add model architecture arguments if provided
    if args.get("hidden_dim"):
        cmd.extend(["--hidden_dim", str(args["hidden_dim"])])
    if args.get("dropout_rate"):
        cmd.extend(["--dropout_rate", str(args["dropout_rate"])])
    if args.get("num_layers"):
        cmd.extend(["--num_layers", str(args["num_layers"])])
    if args.get("layer_ratio"):
        cmd.extend(["--layer_ratio", str(args["layer_ratio"])])
    if args.get("activation"):
        cmd.extend(["--activation", args["activation"]])
    if args.get("use_skip_connection"):
        cmd.append("--use_skip_connection")
    
    start_time = time.time()
    
    try:
        # Run the evaluation
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            logger.info(f"✅ Evaluation completed for {checkpoint_name} in {duration:.2f}s")
            return {
                "checkpoint": checkpoint_name,
                "status": "success",
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        else:
            # Analyze the error to provide better error messages
            error_msg = result.stderr
            print(f"\n=== ERROR OUTPUT FOR {checkpoint_name} ===")
            print(f"Return code: {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            print("="*50)
            
            if "size mismatch" in error_msg or "shape" in error_msg:
                error_type = "architecture_mismatch"
                logger.error(f"🏗️ Architecture mismatch for {checkpoint_name} after {duration:.2f}s")
                logger.error(f"Model architecture doesn't match checkpoint dimensions")
            elif "CUDA out of memory" in error_msg:
                error_type = "out_of_memory"
                logger.error(f"💾 Out of memory for {checkpoint_name} after {duration:.2f}s")
            elif "timeout" in error_msg.lower():
                error_type = "timeout"
                logger.error(f"⏰ Evaluation timed out for {checkpoint_name} after {duration:.2f}s")
            elif "Failed to build exact match model" in error_msg:
                error_type = "incompatible_checkpoint"
                logger.error(f"🔧 Incompatible checkpoint structure for {checkpoint_name} after {duration:.2f}s")
                logger.error(f"Checkpoint has unsupported architecture")
            elif "ModuleNotFoundError" in error_msg or "ImportError" in error_msg:
                error_type = "import_error"
                logger.error(f"📦 Import error for {checkpoint_name} after {duration:.2f}s")
                logger.error(f"Missing dependencies or module not found")
            else:
                error_type = "failed"
                logger.error(f"❌ Evaluation failed for {checkpoint_name} after {duration:.2f}s")
                logger.error(f"Unknown error - check output above for details")
            
            logger.error(f"Error output: {error_msg}")
            return {
                "checkpoint": checkpoint_name,
                "status": error_type,
                "duration": duration,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ Evaluation timed out for {checkpoint_name}")
        return {
            "checkpoint": checkpoint_name,
            "status": "timeout",
            "duration": 3600
        }
    except Exception as e:
        logger.error(f"💥 Unexpected error for {checkpoint_name}: {str(e)}")
        return {
            "checkpoint": checkpoint_name,
            "status": "error",
            "error": str(e)
        }

def run_evaluations_parallel(checkpoints: List[Path], args: Dict[str, Any], max_workers: int) -> List[Dict[str, Any]]:
    """
    Run evaluations in parallel using ProcessPoolExecutor.
    
    Args:
        checkpoints: List of checkpoint paths
        args: Evaluation arguments
        max_workers: Maximum number of parallel workers
    
    Returns:
        List of evaluation results
    """
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all evaluation jobs
        future_to_checkpoint = {
            executor.submit(run_evaluation, checkpoint, args): checkpoint 
            for checkpoint in checkpoints
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_checkpoint):
            checkpoint = future_to_checkpoint[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error getting result for {checkpoint.name}: {str(e)}")
                results.append({
                    "checkpoint": checkpoint.name,
                    "status": "error",
                    "error": str(e)
                })
    
    return results

def run_evaluations_sequential(checkpoints: List[Path], args: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Run evaluations sequentially.
    
    Args:
        checkpoints: List of checkpoint paths
        args: Evaluation arguments
    
    Returns:
        List of evaluation results
    """
    results = []
    
    for i, checkpoint in enumerate(checkpoints, 1):
        logger.info(f"Progress: {i}/{len(checkpoints)} - {checkpoint.name}")
        result = run_evaluation(checkpoint, args)
        results.append(result)
        
        # Small delay between runs to avoid overwhelming the system
        time.sleep(1)
    
    return results

def save_results(results: List[Dict[str, Any]], output_file: str):
    """
    Save evaluation results to a JSON file.
    
    Args:
        results: List of evaluation results
        output_file: Output file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {output_path}")

def print_summary(results: List[Dict[str, Any]]):
    """
    Print a summary of evaluation results.
    
    Args:
        results: List of evaluation results
    """
    total = len(results)
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    architecture_mismatch = sum(1 for r in results if r["status"] == "architecture_mismatch")
    incompatible_checkpoint = sum(1 for r in results if r["status"] == "incompatible_checkpoint")
    out_of_memory = sum(1 for r in results if r["status"] == "out_of_memory")
    import_error = sum(1 for r in results if r["status"] == "import_error")
    timeout = sum(1 for r in results if r["status"] == "timeout")
    error = sum(1 for r in results if r["status"] == "error")
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total checkpoints: {total}")
    print(f"Successful: {successful} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Architecture Mismatch: {architecture_mismatch} 🏗️")
    print(f"Incompatible Checkpoint: {incompatible_checkpoint} 🔧")
    print(f"Out of Memory: {out_of_memory} 💾")
    print(f"Import Error: {import_error} 📦")
    print(f"Timeout: {timeout} ⏰")
    print(f"Error: {error} 💥")
    print("="*60)
    
    if failed > 0 or architecture_mismatch > 0 or incompatible_checkpoint > 0 or out_of_memory > 0 or import_error > 0 or timeout > 0 or error > 0:
        print("\nFailed checkpoints:")
        for result in results:
            if result["status"] != "success":
                status_emoji = {
                    "failed": "❌",
                    "architecture_mismatch": "🏗️",
                    "incompatible_checkpoint": "🔧",
                    "out_of_memory": "💾",
                    "import_error": "📦",
                    "timeout": "⏰",
                    "error": "💥"
                }.get(result["status"], "❓")
                
                print(f"  - {result['checkpoint']}: {result['status']} {status_emoji}")
                if "error" in result:
                    print(f"    Error: {result['error']}")
                elif result["status"] == "architecture_mismatch":
                    print(f"    Issue: Model architecture doesn't match checkpoint dimensions")
                elif result["status"] == "incompatible_checkpoint":
                    print(f"    Issue: Checkpoint has unsupported architecture")
                elif result["status"] == "out_of_memory":
                    print(f"    Issue: GPU memory insufficient for this checkpoint")
                elif result["status"] == "import_error":
                    print(f"    Issue: Missing dependencies or module not found")

def filter_checkpoints_by_previous_results(checkpoints: List[Path], output_file: str, skip_architecture_mismatch: bool = False, skip_incompatible: bool = False) -> List[Path]:
    """
    Filter checkpoints based on previous evaluation results.
    
    Args:
        checkpoints: List of checkpoint paths
        output_file: Path to previous results file
        skip_architecture_mismatch: Whether to skip checkpoints with architecture mismatches
        skip_incompatible: Whether to skip incompatible checkpoints
    
    Returns:
        Filtered list of checkpoint paths
    """
    if not Path(output_file).exists():
        return checkpoints
    
    try:
        with open(output_file, 'r') as f:
            previous_results = json.load(f)
        
        # Create a set of checkpoints to skip
        skip_checkpoints = set()
        
        for result in previous_results:
            checkpoint_name = result["checkpoint"]
            status = result["status"]
            
            if skip_architecture_mismatch and status == "architecture_mismatch":
                skip_checkpoints.add(checkpoint_name)
                logger.info(f"Skipping {checkpoint_name} due to previous architecture mismatch")
            elif skip_incompatible and status == "incompatible_checkpoint":
                skip_checkpoints.add(checkpoint_name)
                logger.info(f"Skipping {checkpoint_name} due to previous incompatibility")
        
        # Filter out checkpoints to skip
        filtered_checkpoints = [
            checkpoint for checkpoint in checkpoints 
            if checkpoint.name not in skip_checkpoints
        ]
        
        logger.info(f"Filtered {len(checkpoints) - len(filtered_checkpoints)} checkpoints based on previous results")
        return filtered_checkpoints
        
    except Exception as e:
        logger.warning(f"Could not load previous results: {e}")
        return checkpoints

def main():
    parser = argparse.ArgumentParser(description="Run evaluation for all diffusion model checkpoints")
    
    # Evaluation parameters
    parser.add_argument('--data_name', type=str, default="ML1M", help="Dataset name")
    parser.add_argument('--recommender_name', type=str, default="VAE", help="Recommender name")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--lambda_cf', type=float, default=0.5, help="Lambda counterfactual")
    parser.add_argument('--lambda_l1', type=float, default=1.0, help="Lambda L1")
    parser.add_argument('--lambda_preserve', type=float, default=1.0, help="Lambda preserve")
    
    # Model architecture parameters
    parser.add_argument('--hidden_dim', type=int, help="Hidden dimension")
    parser.add_argument('--dropout_rate', type=float, help="Dropout rate")
    parser.add_argument('--num_layers', type=int, help="Number of layers")
    parser.add_argument('--layer_ratio', type=float, help="Layer ratio")
    parser.add_argument('--activation', type=str, help="Activation function")
    parser.add_argument('--use_skip_connection', action='store_true', help="Use skip connection")
    
    # Execution options
    parser.add_argument('--parallel', action='store_true', help="Run evaluations in parallel")
    parser.add_argument('--max_workers', type=int, default=4, help="Maximum number of parallel workers")
    parser.add_argument('--filter_pattern', type=str, help="Filter checkpoints by pattern")
    parser.add_argument('--exclude_pattern', type=str, help="Exclude checkpoints by pattern")
    parser.add_argument('--dry_run', action='store_true', help="Show what would be run without actually running")
    parser.add_argument('--output_file', type=str, default="eval_results.json", help="Output file for results")
    parser.add_argument('--skip_architecture_mismatch', action='store_true', help="Skip checkpoints with architecture mismatches")
    parser.add_argument('--skip_incompatible', action='store_true', help="Skip incompatible checkpoints")
    parser.add_argument('--retry_failed', action='store_true', help="Retry failed evaluations with different parameters")
    
    args = parser.parse_args()
    
    # Convert args to dictionary
    args_dict = vars(args)
    
    # Discover checkpoints
    checkpoints_dir = Path("checkpoints/diffusionModels")
    checkpoints = discover_checkpoints(
        checkpoints_dir, 
        filter_pattern=args.filter_pattern,
        exclude_pattern=args.exclude_pattern
    )
    
    if not checkpoints:
        logger.error("No checkpoints found!")
        return
    
    # Print what will be run
    print(f"\nFound {len(checkpoints)} checkpoints to evaluate:")
    for checkpoint in checkpoints:
        print(f"  - {checkpoint.name}")
    
    if args.dry_run:
        print("\nDry run mode - no evaluations will be performed")
        return
    
    # Confirm before proceeding
    # response = input(f"\nProceed with evaluating {len(checkpoints)} checkpoints? (y/N): ")
    response = 'y'
    if response.lower() != 'y':
        print("Evaluation cancelled")
        return
    
    # Filter checkpoints based on previous results if requested
    if args.skip_architecture_mismatch or args.skip_incompatible:
        checkpoints = filter_checkpoints_by_previous_results(
            checkpoints, 
            args.output_file, 
            skip_architecture_mismatch=args.skip_architecture_mismatch,
            skip_incompatible=args.skip_incompatible
        )
    
    # Run evaluations
    start_time = time.time()
    
    if args.parallel:
        logger.info(f"Running evaluations in parallel with {args.max_workers} workers")
        results = run_evaluations_parallel(checkpoints, args_dict, args.max_workers)
    else:
        logger.info("Running evaluations sequentially")
        results = run_evaluations_sequential(checkpoints, args_dict)
    
    total_time = time.time() - start_time
    
    # Save results
    save_results(results, args.output_file)
    
    # Print summary
    print_summary(results)
    
    logger.info(f"Total execution time: {total_time:.2f}s")

if __name__ == "__main__":
    main() 