#!/usr/bin/env python3
"""
Debug script to understand the ICSI dataset meeting distribution
This works from the masters-thesis directory and references the treeseg implementation
"""
import os
import sys

def check_meetings():
    # Path to the treeseg implementation's ICSI dataset
    treeseg_dir = "../treeseg"
    anno_dir = os.path.join(treeseg_dir, "datasets", "ICSI", "Contributions", "TopicSegmentation")
    
    print(f"Looking for meetings in: {anno_dir}")
    
    if not os.path.exists(anno_dir):
        print(f"Directory {anno_dir} not found!")
        return
    
    topic_fnms = os.listdir(anno_dir)
    meetings = sorted(
        [fnm.split(".")[0] for fnm in topic_fnms if fnm.endswith(".xml")]
    )
    
    print(f"Total meetings found: {len(meetings)}")
    print(f"First 10 meetings: {meetings[:10]}")
    print(f"Last 10 meetings: {meetings[-10:]}")
    
    # Simulate the current fold splitting logic from treeseg/datasets/icsi.py
    dev_meetings = meetings[:5]
    test_meetings = meetings[5:]
    
    print(f"\nCurrent treeseg implementation fold splitting:")
    print(f"Dev fold (first 5): {dev_meetings}")
    print(f"Test fold (remaining {len(test_meetings)}): {test_meetings[:10]}...")
    
    print(f"\nAnalysis:")
    print(f"- Total available meetings: {len(meetings)}")
    print(f"- Dev fold gets: {len(dev_meetings)} meetings")
    print(f"- Test fold gets: {len(test_meetings)} meetings")
    print(f"- This matches the research paper's claim of 75+ meetings!")
    
    print(f"\nThe issue is in the fold splitting logic:")
    print(f"- The code uses meetings[:5] for dev and meetings[5:] for test")
    print(f"- This gives only 5 meetings for dev and {len(test_meetings)} for test")
    print(f"- If you want to use more meetings, you need to modify the splitting logic")
    
    # Show what a more balanced split might look like
    mid_point = len(meetings) // 2
    balanced_dev = meetings[:mid_point]
    balanced_test = meetings[mid_point:]
    
    print(f"\nAlternative balanced split (50/50):")
    print(f"Dev fold: {len(balanced_dev)} meetings")
    print(f"Test fold: {len(balanced_test)} meetings")

if __name__ == "__main__":
    check_meetings()