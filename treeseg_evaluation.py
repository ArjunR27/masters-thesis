#!/usr/bin/env python3

import os
import sys
import json
import numpy as np
import structlog
from datetime import datetime
from typing import Dict, List, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial

sys.path.append(os.path.dirname(__file__))

from treeseg.treeseg import TreeSeg
from configs import treeseg_configs
from datasets import ICSIDataset, AMIDataset

logger = structlog.get_logger("treeseg_evaluation")

EMBEDDING_MODELS = [
    # "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-large-en-v1.5"
]


DATASETS = ["ami", "icsi"]


MAX_MEETINGS_PER_DATASET = 25  # Set to 2 or 3 for quick testing


USE_PARALLEL = True

NUM_WORKERS = None 


def window_diff(ref, hyp, k=None):
    """Calculate WindowDiff metric (lower is better)."""
    if len(ref) != len(hyp):
        raise ValueError("Reference and hypothesis must have same length")

    ref = list(map(int, ref))
    hyp = list(map(int, hyp))

    if k is None:
        k = int(round(len(ref) / (sum(ref) + 1) / 2.0))
    k = max(1, min(k, len(ref) - 1))

    ref_cum = np.cumsum([0] + ref)
    hyp_cum = np.cumsum([0] + hyp)

    errors = 0
    windows = len(ref) - k + 1
    for i in range(windows):
        ref_boundaries = ref_cum[i + k] - ref_cum[i]
        hyp_boundaries = hyp_cum[i + k] - hyp_cum[i]
        if ref_boundaries != hyp_boundaries:
            errors += 1

    return errors / windows


def pk_metric(ref, hyp, k=None):
    if len(ref) != len(hyp):
        raise ValueError("Reference and hypothesis must have same length")

    ref = list(map(int, ref))
    hyp = list(map(int, hyp))

    if k is None:
        k = int(round(len(ref) / (sum(ref) + 1) / 2.0))
    k = max(1, min(k, len(ref) - 1))

    ref_cum = np.cumsum([0] + ref)
    hyp_cum = np.cumsum([0] + hyp)

    errors = 0
    pairs = len(ref) - k + 1
    for i in range(pairs):
        ref_same = (ref_cum[i + k] - ref_cum[i]) == 0
        hyp_same = (hyp_cum[i + k] - hyp_cum[i]) == 0
        if ref_same != hyp_same:
            errors += 1

    return errors / pairs


def evaluate_meeting(model, entries, transitions):
    true_K = int(sum(transitions)) + 1
    transitions_hat = model.segment_meeting(true_K)
    
    wd_score = window_diff(transitions, transitions_hat)
    pk_score = pk_metric(transitions, transitions_hat)
    
    return {
        'window_diff': wd_score,
        'pk': pk_score,
        'true_boundaries': int(sum(transitions)),
        'pred_boundaries': int(sum(transitions_hat)),
    }


def evaluate_single_meeting_parallel(args):
    meeting_idx, meeting_name, entries, transitions, config = args
    
    try:
        # Force CPU usage in parallel mode to avoid GPU memory exhaustion
        parallel_config = config.copy()
        parallel_config['HF_DEVICE'] = 'cpu'
        
        # Create model for this meeting
        model = TreeSeg(configs=parallel_config, entries=entries)
        result = evaluate_meeting(model, entries, transitions)
        result['meeting'] = meeting_name
        result['index'] = meeting_idx
        return result
    except Exception as e:
        return {
            'meeting': meeting_name,
            'index': meeting_idx,
            'error': str(e),
            'window_diff': None,
            'pk': None,
        }


def evaluate_dataset_with_model(dataset_name: str, embedding_model: str) -> Dict:
    logger.info(f"Evaluating {dataset_name.upper()} with {embedding_model}")
    
    # Load dataset
    if dataset_name == "ami":
        dataset = AMIDataset(fold='test')
    else:
        dataset = ICSIDataset(fold='test')
    
    dataset.load_dataset()
    
    # Apply meeting limit if set
    meetings = dataset.meetings
    if MAX_MEETINGS_PER_DATASET is not None:
        meetings = meetings[:MAX_MEETINGS_PER_DATASET]
        logger.info(f"Limited to {len(meetings)}/{len(dataset.meetings)} meetings for testing")
    else:
        logger.info(f"Loaded {len(meetings)} test meetings")
    
    # Create config with this embedding model
    config = treeseg_configs[dataset_name].copy()
    config['HF_EMBEDDING_MODEL'] = embedding_model
    
    meeting_results = []
    
    if USE_PARALLEL:
        # Parallel evaluation
        num_workers = NUM_WORKERS if NUM_WORKERS else min(cpu_count() - 1, 4)
        num_workers = max(1, num_workers)  # At least 1 worker
        
        logger.info(f"  Using {num_workers} parallel workers")
        
        # Prepare arguments for parallel processing
        eval_args = []
        for i, meeting in enumerate(meetings):
            entries = dataset.notes[meeting]
            transitions = dataset.transitions[meeting]
            eval_args.append((i, meeting, entries, transitions, config))
        
        # Run in parallel
        with Pool(num_workers) as pool:
            meeting_results = pool.map(evaluate_single_meeting_parallel, eval_args)
        
        # Log results
        for result in meeting_results:
            if 'error' in result:
                logger.error(f"  Meeting {result['index']+1}: {result['meeting']} - FAILED: {result['error']}")
            elif result['window_diff'] is not None:
                logger.info(f"  Meeting {result['index']+1}: {result['meeting']} - WD: {result['window_diff']:.4f}, Pk: {result['pk']:.4f}")
        
        # Filter out failed results
        meeting_results = [r for r in meeting_results if r.get('window_diff') is not None]
        
    else:
        # Sequential evaluation (original behavior)
        for i, meeting in enumerate(meetings):
            logger.info(f"  Meeting {i+1}/{len(meetings)}: {meeting}")
            
            entries = dataset.notes[meeting]
            transitions = dataset.transitions[meeting]
            
            try:
                # Run TreeSeg
                model = TreeSeg(configs=config, entries=entries)
                result = evaluate_meeting(model, entries, transitions)
                meeting_results.append(result)
                
                logger.info(f"    WD: {result['window_diff']:.4f}, Pk: {result['pk']:.4f}")
                
            except Exception as e:
                logger.error(f"    Failed: {e}")
                continue
    
    # Compute averages
    if not meeting_results:
        return {
            'avg_window_diff': None,
            'avg_pk': None,
            'num_meetings': 0,
        }
    
    avg_wd = np.mean([r['window_diff'] for r in meeting_results])
    avg_pk = np.mean([r['pk'] for r in meeting_results])
    
    return {
        'avg_window_diff': avg_wd,
        'avg_pk': avg_pk,
        'std_window_diff': np.std([r['window_diff'] for r in meeting_results]),
        'std_pk': np.std([r['pk'] for r in meeting_results]),
        'num_meetings': len(meeting_results),
        'meeting_results': meeting_results,
    }


def format_model_name(model_path: str) -> str:
    """Shorten model name for display."""
    if "/" in model_path:
        return model_path.split("/")[1]
    return model_path


def print_results_table(results: Dict):
    """Print results in a nicely formatted table."""
    print("\n" + "=" * 120)
    print("TreeSeg Embedding Model Comparison - Test Set Results")
    print("=" * 120)
    print()
    
    print(f"Evaluation completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    header = f"{'Model':<40} | {'Dataset':<8} | {'Meetings':<8} | {'WindowDiff':<20} | {'Pk':<20}"
    print(header)
    print("-" * 120)
    
    for model in EMBEDDING_MODELS:
        short_name = format_model_name(model)
        for dataset in DATASETS:
            key = f"{model}_{dataset}"
            if key not in results:
                continue
            
            r = results[key]
            if r['num_meetings'] == 0:
                wd_str = "FAILED"
                pk_str = "FAILED"
            else:
                wd_str = f"{r['avg_window_diff']:.4f} ± {r['std_window_diff']:.4f}"
                pk_str = f"{r['avg_pk']:.4f} ± {r['std_pk']:.4f}"
            
            row = f"{short_name:<40} | {dataset.upper():<8} | {r['num_meetings']:<8} | {wd_str:<20} | {pk_str:<20}"
            print(row)
        print("-" * 120)
    
    print()
    print("Note: Lower scores are better for both WindowDiff and Pk metrics.")
    print("=" * 120)
    print()
    
    print("\nSummary Statistics:")
    print("-" * 80)
    
    for dataset in DATASETS:
        print(f"\n{dataset.upper()} Dataset:")
        dataset_results = [(model, results[f"{model}_{dataset}"]) 
                          for model in EMBEDDING_MODELS 
                          if f"{model}_{dataset}" in results and results[f"{model}_{dataset}"]['num_meetings'] > 0]
        
        if not dataset_results:
            print("  No successful evaluations")
            continue
        
        # Best WindowDiff
        best_wd = min(dataset_results, key=lambda x: x[1]['avg_window_diff'])
        print(f"  Best WindowDiff: {format_model_name(best_wd[0]):<40} ({best_wd[1]['avg_window_diff']:.4f})")
        
        # Best Pk
        best_pk = min(dataset_results, key=lambda x: x[1]['avg_pk'])
        print(f"  Best Pk:         {format_model_name(best_pk[0]):<40} ({best_pk[1]['avg_pk']:.4f})")
    
    print()


def save_results_json(results: Dict, output_file: str = "treeseg_evaluation_results.json"):
    """Save detailed results to JSON file."""
    # Convert to serializable format
    serializable_results = {}
    for key, value in results.items():
        serializable_results[key] = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in value.items()
            if k != 'meeting_results'
        }
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'models': EMBEDDING_MODELS,
            'datasets': DATASETS,
            'results': serializable_results,
        }, f, indent=2)
    
    logger.info(f"Detailed results saved to {output_file}")


def main():
    """Main evaluation pipeline."""
    print("\n" + "=" * 120)
    print("TreeSeg Embedding Model Evaluation")
    print("=" * 120)
    print(f"\nEvaluating {len(EMBEDDING_MODELS)} embedding models on {len(DATASETS)} datasets (test split)")
    print(f"Total evaluations: {len(EMBEDDING_MODELS) * len(DATASETS)}")
    
    print("\nOptimization Settings:")
    print(f"  Parallel Processing: {'ENABLED' if USE_PARALLEL else 'DISABLED'}")
    if USE_PARALLEL:
        num_workers = NUM_WORKERS if NUM_WORKERS else min(cpu_count() - 1, 4)
        print(f"  Number of Workers: {max(1, num_workers)} (CPU cores: {cpu_count()})")
        print(f"  Device: CPU (forced in parallel mode to avoid GPU memory issues)")
    else:
        import torch
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {device} (GPU-accelerated if available)")
    if MAX_MEETINGS_PER_DATASET:
        print(f"  Max Meetings per Dataset: {MAX_MEETINGS_PER_DATASET} (limited for testing)")
    else:
        print(f"  Max Meetings per Dataset: ALL (full evaluation)")
    
    print("\nModels to evaluate:")
    for i, model in enumerate(EMBEDDING_MODELS, 1):
        print(f"  {i}. {model}")
    print("\nDatasets: AMI (test), ICSI (test)")
    print("=" * 120)
    
    results = {}
    total_evals = len(EMBEDDING_MODELS) * len(DATASETS)
    current_eval = 0
    
    # Evaluate each combination
    for model in EMBEDDING_MODELS:
        for dataset in DATASETS:
            current_eval += 1
            print(f"\n{'='*120}")
            print(f"Progress: [{current_eval}/{total_evals}] Evaluating {format_model_name(model)} on {dataset.upper()}")
            print(f"{'='*120}")
            
            try:
                result = evaluate_dataset_with_model(dataset, model)
                results[f"{model}_{dataset}"] = result
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                results[f"{model}_{dataset}"] = {
                    'avg_window_diff': None,
                    'avg_pk': None,
                    'num_meetings': 0,
                }
    
    print_results_table(results)
    
    save_results_json(results)
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()

