#!/usr/bin/env python3
"""
Test script for the datasets directory functionality.
This script demonstrates how to load, process, and inspect the ICSI dataset.
"""

import sys
import os
import json
from pathlib import Path

# Add the datasets directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'datasets'))

from datasets import ICSIDataset

def test_dataset_loading():
    """Test basic dataset loading functionality."""
    print("=" * 60)
    print("TESTING DATASET LOADING")
    print("=" * 60)
    
    try:
        # Initialize dataset for development fold
        print("Initializing ICSI dataset (dev fold)...")
        dataset = ICSIDataset(fold="dev")
        
        print(f"✓ Dataset initialized successfully")
        print(f"  - Meetings: {dataset.meetings}")
        print(f"  - Speakers: {dataset.speakers}")
        print(f"  - Min segment size: {dataset.MIN_SEGMENT_SIZE}")
        
        return dataset
        
    except Exception as e:
        print(f"✗ Error initializing dataset: {e}")
        return None

def test_asset_loading(dataset):
    """Test loading of words and utterances."""
    print("\n" + "=" * 60)
    print("TESTING ASSET LOADING")
    print("=" * 60)
    
    try:
        print("Loading assets (words and utterances)...")
        dataset.load_assets()
        
        print("✓ Assets loaded successfully")
        
        # Check first meeting
        first_meeting = dataset.meetings[0]
        print(f"\nInspecting first meeting: {first_meeting}")
        
        if first_meeting in dataset.words:
            speakers_with_words = list(dataset.words[first_meeting].keys())
            print(f"  - Speakers with word data: {speakers_with_words}")
            
            # Check word count for first speaker
            if speakers_with_words:
                first_speaker = speakers_with_words[0]
                word_count = len(dataset.words[first_meeting][first_speaker]["words"])
                print(f"  - Words for {first_speaker}: {word_count}")
                
                # Show first few words
                words = dataset.words[first_meeting][first_speaker]["words"][:5]
                print(f"  - First 5 words: {[w['text'] if w else 'None' for w in words]}")
        
        if first_meeting in dataset.index:
            speakers_with_utterances = list(dataset.index[first_meeting].keys())
            print(f"  - Speakers with utterance data: {speakers_with_utterances}")
            
            # Count utterances for first speaker
            if speakers_with_utterances:
                first_speaker = speakers_with_utterances[0]
                utterance_count = len(dataset.index[first_meeting][first_speaker])
                print(f"  - Utterances for {first_speaker}: {utterance_count}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading assets: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_annotation_loading(dataset):
    """Test loading of topic annotations."""
    print("\n" + "=" * 60)
    print("TESTING ANNOTATION LOADING")
    print("=" * 60)
    
    try:
        print("Loading annotation trees...")
        dataset.load_anno_trees()
        
        print("✓ Annotation trees loaded successfully")
        
        # Check first meeting
        first_meeting = dataset.meetings[0]
        if first_meeting in dataset.anno_roots:
            root = dataset.anno_roots[first_meeting]
            print(f"\nInspecting annotation tree for {first_meeting}:")
            print(f"  - Root path: {root.path}")
            print(f"  - Root tag: {root.tag}")
            print(f"  - Is leaf: {root.is_leaf}")
            print(f"  - Number of children: {len(root.nn)}")
            
            # Show first few children
            if root.nn:
                print(f"  - First child: {root.nn[0].path} ({root.nn[0].tag})")
                if len(root.nn) > 1:
                    print(f"  - Second child: {root.nn[1].path} ({root.nn[1].tag})")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading annotations: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_dataset_loading(dataset):
    """Test complete dataset loading and processing."""
    print("\n" + "=" * 60)
    print("TESTING FULL DATASET LOADING")
    print("=" * 60)
    
    try:
        print("Loading complete dataset...")
        dataset.load_dataset()
        
        print("✓ Full dataset loaded successfully")
        
        # Check first meeting
        first_meeting = dataset.meetings[0]
        print(f"\nInspecting processed data for {first_meeting}:")
        
        if first_meeting in dataset.notes:
            note_count = len(dataset.notes[first_meeting])
            print(f"  - Total conversation entries: {note_count}")
            
            # Show first few entries
            if note_count > 0:
                print(f"  - First entry: {dataset.notes[first_meeting][0]['composite']}")
                if note_count > 1:
                    print(f"  - Second entry: {dataset.notes[first_meeting][1]['composite']}")
        
        if first_meeting in dataset.transitions:
            transition_count = len(dataset.transitions[first_meeting])
            boundary_count = sum(dataset.transitions[first_meeting])
            print(f"  - Total transitions: {transition_count}")
            print(f"  - Boundary points: {boundary_count}")
        
        if first_meeting in dataset.hier_transitions:
            hier_levels = len(dataset.hier_transitions[first_meeting])
            print(f"  - Hierarchical levels: {hier_levels}")
            
            # Show boundary counts for each level
            for i, transitions in enumerate(dataset.hier_transitions[first_meeting]):
                boundary_count = sum(transitions)
                print(f"    Level {i}: {boundary_count} boundaries")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading full dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tree_inspection(dataset):
    """Test annotation tree inspection utilities."""
    print("\n" + "=" * 60)
    print("TESTING TREE INSPECTION")
    print("=" * 60)
    
    try:
        first_meeting = dataset.meetings[0]
        root = dataset.anno_roots[first_meeting]
        
        print(f"Inspecting annotation tree for {first_meeting}:")
        
        # Discover leaves
        leaves = dataset.discover_anno_leaves(root)
        print(f"  - Total leaf nodes: {len(leaves)}")
        
        # Show first few leaves
        for i, leaf in enumerate(leaves[:3]):
            print(f"  - Leaf {i+1}: {leaf.path}")
            print(f"    Keys: {len(leaf.keys)}")
            print(f"    Conversation entries: {len(leaf.convo)}")
            if leaf.convo:
                print(f"    First entry: {leaf.convo[0]['composite']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error inspecting tree: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_timed_utterances():
    """Test dataset with timed utterances enabled."""
    print("\n" + "=" * 60)
    print("TESTING TIMED UTTERANCES")
    print("=" * 60)
    
    try:
        print("Initializing dataset with timed utterances...")
        dataset = ICSIDataset(fold="dev", timed_utterances=True)
        dataset.load_dataset()
        
        print("✓ Timed dataset loaded successfully")
        
        # Check first meeting
        first_meeting = dataset.meetings[0]
        if first_meeting in dataset.notes and dataset.notes[first_meeting]:
            first_entry = dataset.notes[first_meeting][0]
            print(f"  - First entry with timing: {first_entry['composite']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error with timed utterances: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests in sequence."""
    print("ICSI DATASET TESTING SUITE")
    print("=" * 60)
    
    # Test 1: Basic initialization
    dataset = test_dataset_loading()
    if not dataset:
        print("\n❌ Basic initialization failed. Cannot continue.")
        return False
    
    # Test 2: Asset loading
    if not test_asset_loading(dataset):
        print("\n❌ Asset loading failed.")
        return False
    
    # Test 3: Annotation loading
    if not test_annotation_loading(dataset):
        print("\n❌ Annotation loading failed.")
        return False
    
    # Test 4: Full dataset loading
    if not test_full_dataset_loading(dataset):
        print("\n❌ Full dataset loading failed.")
        return False
    
    # Test 5: Tree inspection
    if not test_tree_inspection(dataset):
        print("\n❌ Tree inspection failed.")
        return False
    
    # Test 6: Timed utterances
    if not test_timed_utterances():
        print("\n❌ Timed utterances test failed.")
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
