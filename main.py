#!/usr/bin/env python3
# pyright: reportUnreachable=false

"""
Simplified main.py to test treeseg implementation with Hugging Face embeddings.
Runs directly with: python main.py
"""

import os
import sys
import json
import numpy as np
import structlog

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from treeseg.treeseg import TreeSeg
from configs import treeseg_configs
from datasets import ICSIDataset

logger = structlog.get_logger("main")

def window_diff(ref, hyp, k=None):
    if len(ref) != len(hyp):
        raise ValueError("Reference and hypothesis must have same length")

    # Accept numpy arrays or lists of 0/1
    ref = list(map(int, ref))
    hyp = list(map(int, hyp))

    # Default k: half the average segment size (same as original script)
    if k is None:
        k = int(round(len(ref) / (sum(ref) + 1) / 2.0))
    k = max(1, min(k, len(ref) - 1))  # keep k in a sane range

    ref_cum = np.cumsum([0] + ref)
    hyp_cum = np.cumsum([0] + hyp)

    errors = 0
    windows = len(ref) - k + 1
    for i in range(windows):
        ref_boundaries = ref_cum[i + k] - ref_cum[i]
        hyp_boundaries = hyp_cum[i + k] - hyp_cum[i]
        # NLTK windowdiff increments when the counts differ (not by how much)
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
        # Same-segment test: no boundary between i and i+k
        ref_same = (ref_cum[i + k] - ref_cum[i]) == 0
        hyp_same = (hyp_cum[i + k] - hyp_cum[i]) == 0
        if ref_same != hyp_same:
            errors += 1

    return errors / pairs

def evaluate_with_tolerance(transitions, transitions_hat, tolerance=2):
    """
    Evaluate with tolerance window (allows boundaries to be off by N utterances).
    
    Returns:
        precision, recall, f1
    """
    true_boundaries = [i for i, t in enumerate(transitions) if t == 1]
    pred_boundaries = [i for i, t in enumerate(transitions_hat) if t == 1]
    
    if not pred_boundaries and not true_boundaries:
        return 1.0, 1.0, 1.0
    if not pred_boundaries:
        return 0.0, 0.0, 0.0
    if not true_boundaries:
        return 0.0, 0.0, 0.0
    
    matched_true = set()
    matched_pred = set()
    
    for pred_idx in pred_boundaries:
        for true_idx in true_boundaries:
            if abs(pred_idx - true_idx) <= tolerance and true_idx not in matched_true:
                matched_true.add(true_idx)
                matched_pred.add(pred_idx)
                break
            
    
    precision = len(matched_pred) / len(pred_boundaries)
    recall = len(matched_true) / len(true_boundaries)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def main():
    """Run treeseg on ICSI dataset with Hugging Face embeddings."""
    
    print("=" * 60)
    print("Testing TreeSeg with Hugging Face Embeddings")
    print("=" * 60)
    
    # Configuration
    DATASET = "icsi"  # Use ICSI dataset
    FOLD = "dev"      # Use dev fold
    MEETING_INDEX = 0 # Test on first meeting
    
    logger.info(f"Dataset: {DATASET}")
    logger.info(f"Fold: {FOLD}")
    logger.info(f"Meeting index: {MEETING_INDEX}")
    
    # Load dataset
    print("\n1. Loading dataset...")
    try:
        dataset = ICSIDataset(fold=FOLD)
        dataset.load_dataset()
        logger.info(f"Loaded {len(dataset.meetings)} meetings")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # Get target meeting
    if MEETING_INDEX >= len(dataset.meetings):
        logger.error(f"Meeting index {MEETING_INDEX} out of range. Available: {len(dataset.meetings)}")
        return
    
    meeting = dataset.meetings[MEETING_INDEX]
    logger.info(f"Processing meeting: {meeting}")
    
    # Get meeting data
    entries = dataset.notes[meeting]
    transitions = dataset.transitions[meeting]
    
    logger.info(f"Meeting has {len(entries)} utterances")
    logger.info(f"Ground truth has {sum(transitions)} topic boundaries")
    
    # Show first few utterances
    print("\n2. Sample utterances:")
    for i, entry in enumerate(entries[:3]):
        print(f"   {i+1}. {entry['composite']}")
    if len(entries) > 3:
        print(f"   ... and {len(entries) - 3} more")
    
    # Run TreeSeg
    print("\n3. Running TreeSeg...")
    try:
        # Get config for this dataset
        config = treeseg_configs[DATASET]
        logger.info(f"Using config: {config['HF_EMBEDDING_MODEL']} on {config['HF_DEVICE']}")
        
        # Initialize and run TreeSeg
        model = TreeSeg(configs=config, entries=entries)
        
        # Segment the meeting
        true_K = int(sum(transitions)) + 1
        logger.info(f"Target number of segments: {true_K}")
        
        transitions_hat = model.segment_meeting(true_K)
        
        logger.info(f"Predicted {sum(transitions_hat)} topic boundaries")
        
    except Exception as e:
        logger.error(f"TreeSeg failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Evaluation with multiple metrics
    print("\n4. Evaluation Results:")
    print("=" * 60)
    
    # WindowDiff (lower is better)
    wd_score = window_diff(transitions, transitions_hat)
    print(f"\n   WindowDiff: {wd_score:.4f} (lower is better, 0=perfect)")
    
    # Pk metric (lower is better)
    pk_score = pk_metric(transitions, transitions_hat)
    print(f"   Pk:         {pk_score:.4f} (lower is better, 0=perfect)")
    
    # Tolerance-based metrics
    print("\n   With tolerance windows:")
    for tolerance in [0, 1, 2, 3]:
        prec, rec, f1 = evaluate_with_tolerance(transitions, transitions_hat, tolerance=tolerance)
        print(f"   ±{tolerance} utterances: P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}")
    
    # Basic counts
    print(f"\n   Ground truth boundaries: {sum(transitions)}")
    print(f"   Predicted boundaries:    {sum(transitions_hat)}")
    
    # Show segments
    print("\n5. Predicted Segments:")
    print("=" * 60)
    segment_start = 0
    segment_num = 1
    
    for i in range(len(transitions_hat)):
        if transitions_hat[i] == 1 or i == len(transitions_hat) - 1:
            segment_end = i + 1 if i == len(transitions_hat) - 1 else i
            segment_entries = entries[segment_start:segment_end]
            
            if len(segment_entries) > 0:
                # Check if this aligns with ground truth
                has_true_boundary = any(transitions[j] == 1 for j in range(segment_start, min(segment_end + 3, len(transitions))))
                marker = "✓" if has_true_boundary else " "
                
                print(f"\n   {marker} Segment {segment_num}:")
                print(f"      Utterances {segment_start}-{segment_end-1} ({len(segment_entries)} utterances)")
                
                # Show first utterance of segment
                if len(segment_entries) > 0:
                    first_text = segment_entries[0]['composite'][:80]
                    print(f"      Start: {first_text}...")
                
                segment_num += 1
                segment_start = i + 1 if transitions_hat[i] == 1 else segment_end
    
    print("\n" + "=" * 60)
    print("TreeSeg evaluation completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()