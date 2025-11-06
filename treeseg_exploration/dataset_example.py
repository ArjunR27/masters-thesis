#!/usr/bin/env python3
"""
Simple example showing how to use the ICSI dataset.
This demonstrates basic usage patterns for the dataset.
"""
# pyright: reportUnreachable=false
import sys
import os

# Add the datasets directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'datasets'))

from datasets import ICSIDataset

def basic_usage_example():
    """Basic example of using the ICSI dataset."""
    print("ICSI Dataset - Basic Usage Example")
    print("=" * 40)
    
    # 1. Initialize the dataset
    print("1. Initializing dataset...")
    dataset = ICSIDataset(fold="dev")  # Use "test" for test fold
    
    # 2. Load the complete dataset
    print("2. Loading dataset...")
    dataset.load_dataset()
    
    # 3. Access the data
    print("3. Accessing data...")
    
    # Get list of meetings
    meetings = dataset.meetings
    print(f"   Available meetings: {meetings}")
    
    # Get data for first meeting
    first_meeting = meetings[0]
    print(f"\n   Analyzing meeting: {first_meeting}")
    
    # Get conversation notes (transcript)
    notes = dataset.notes[first_meeting]
    print(f"   Total utterances: {len(notes)}")
    
    # Show first few utterances
    print("\n   First 5 utterances:")
    for i, note in enumerate(notes[:5]):
        print(f"   {i+1}. {note['composite']}")
    
    # Get topic transitions
    transitions = dataset.transitions[first_meeting]
    boundary_count = sum(transitions)
    print(f"\n   Topic boundaries: {boundary_count}")
    print(f"   Total utterances: {len(transitions)}")
    
    # Get hierarchical transitions (multiple granularity levels)
    hier_transitions = dataset.hier_transitions[first_meeting]
    print(f"   Hierarchical levels: {len(hier_transitions)}")
    
    for level, transitions in enumerate(hier_transitions):
        boundaries = sum(transitions)
        print(f"     Level {level}: {boundaries} boundaries")

def advanced_usage_example():
    """Advanced example showing tree inspection and custom processing."""
    print("\n\nICSI Dataset - Advanced Usage Example")
    print("=" * 40)
    
    # Initialize with timed utterances
    dataset = ICSIDataset(fold="dev", timed_utterances=True)
    dataset.load_dataset()
    
    first_meeting = dataset.meetings[0]
    
    # Inspect the annotation tree
    print("1. Annotation Tree Inspection:")
    root = dataset.anno_roots[first_meeting]
    
    # Get all leaf nodes (topic segments)
    leaves = dataset.discover_anno_leaves(root)
    print(f"   Total topic segments: {len(leaves)}")
    
    # Show first few segments
    print("\n   First 3 topic segments:")
    for i, leaf in enumerate(leaves[:3]):
        print(f"   Segment {i+1}: {leaf.path}")
        print(f"     Utterances: {len(leaf.convo)}")
        print(f"     First utterance: {leaf.convo[0]['composite']}")
        print()
    
    # Analyze hierarchical structure
    print("2. Hierarchical Analysis:")
    for level, transitions in enumerate(dataset.hier_transitions[first_meeting]):
        boundaries = sum(transitions)
        total_utterances = len(transitions)
        avg_segment_size = total_utterances / (boundaries + 1) if boundaries > 0 else total_utterances
        
        print(f"   Level {level}: {boundaries} boundaries, avg segment size: {avg_segment_size:.1f}")

def custom_processing_example():
    """Example of custom data processing."""
    print("\n\nICSI Dataset - Custom Processing Example")
    print("=" * 40)
    
    dataset = ICSIDataset(fold="dev")
    dataset.load_dataset()
    
    first_meeting = dataset.meetings[0]
    
    # Extract speaker statistics
    print("1. Speaker Statistics:")
    speaker_stats = {}
    
    for note in dataset.notes[first_meeting]:
        speaker = note['speaker']
        if speaker not in speaker_stats:
            speaker_stats[speaker] = {'count': 0, 'total_chars': 0}
        
        speaker_stats[speaker]['count'] += 1
        speaker_stats[speaker]['total_chars'] += len(note['text'])
    
    for speaker, stats in speaker_stats.items():
        avg_length = stats['total_chars'] / stats['count']
        print(f"   {speaker}: {stats['count']} utterances, avg length: {avg_length:.1f} chars")
    
    # Analyze topic segment sizes
    print("\n2. Topic Segment Analysis:")
    leaves = dataset.discover_anno_leaves(dataset.anno_roots[first_meeting])
    
    segment_sizes = [len(leaf.convo) for leaf in leaves]
    print(f"   Total segments: {len(segment_sizes)}")
    print(f"   Average segment size: {sum(segment_sizes) / len(segment_sizes):.1f} utterances")
    print(f"   Min segment size: {min(segment_sizes)}")
    print(f"   Max segment size: {max(segment_sizes)}")

if __name__ == "__main__":
    try:
        basic_usage_example()
        advanced_usage_example()
        custom_processing_example()
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
