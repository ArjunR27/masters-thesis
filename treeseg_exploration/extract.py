import sys
import os

# Add the datasets directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'datasets'))
from configs import treeseg_configs

from datasets import ICSIDataset
from treeseg.treeseg import TreeSeg

def extract_timestamp_and_text(dataset, meeting_id):
    segments = []

    root = dataset.anno_roots[meeting_id]
    leaves = dataset.discover_anno_leaves(root)

    for leaf in leaves:
        convo = leaf.convo
        if not convo:
            continue
        
        start_time = convo[0]['start']
        end_time = convo[-1]['end']

        spoken_text = "\n".join(utt['composite'] for utt in convo)

        level = leaf.path.count('.')
        
        segments.append({
            'start_time': start_time,
            'end_time': end_time,
            'text': spoken_text,
            'path': leaf.path,
            'level': level
        })

    return segments


def extract_timestamp_and_text_treeseg(dataset, meeting_id, K, configs):
    meeting_notes = dataset.notes[meeting_id]
    
    tree_seg = TreeSeg(configs=configs, entries=meeting_notes)
    tree_seg.segment_meeting(K=K)
    
    segments = []
    
    for leaf in tree_seg.leaves:
        segment_utts = [meeting_notes[idx] for idx in leaf.segment]
        
        if not segment_utts:
            continue
        
        start_time = segment_utts[0]['start']
        end_time = segment_utts[-1]['end']
        spoken_text = "\n".join(utt['composite'] for utt in segment_utts)
        
        segments.append({
            'start_time': start_time,
            'end_time': end_time,
            'text': spoken_text,
            'identifier': leaf.identifier,
            'num_utterances': len(segment_utts)
        })
    
    return segments


def write_segments_to_files(segments, meeting_id, output_dir="output_segments"):
    
    meeting_dir = os.path.join(output_dir, meeting_id)
    os.makedirs(meeting_dir, exist_ok=True)
    
    for i, seg in enumerate(segments, start=1):
        filename = f"segment_{i:03d}.txt"
        filepath = os.path.join(meeting_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Segment {i}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Identifier: {seg.get('identifier', seg.get('path', 'N/A'))}\n")
            f.write(f"Start Time: {seg['start_time']:.2f}s\n")
            f.write(f"End Time: {seg['end_time']:.2f}s\n")
            f.write(f"Duration: {seg['end_time'] - seg['start_time']:.2f}s\n")
            
            if 'num_utterances' in seg:
                f.write(f"Number of Utterances: {seg['num_utterances']}\n")
            if 'level' in seg:
                f.write(f"Level: {seg['level']}\n")
            
            f.write(f"{'='*60}\n\n")
            
            f.write(seg['text'])
            f.write("\n")
    
    print(f"\nWrote {len(segments)} segments to {meeting_dir}/")


def main():
    dataset = ICSIDataset(fold='dev')
    dataset.load_dataset()
    meetings = dataset.meetings
    first_meeting = meetings[4]
    
    print("Extracting ICSI ground truth segments...")
    icsi_segments = extract_timestamp_and_text(dataset, first_meeting)
    print(f"Total ICSI segments: {len(icsi_segments)}")
    
    write_segments_to_files(icsi_segments, f"{first_meeting}_icsi", "output_segments")
    
    print("\n" + "=" * 60)
    print("TreeSeg Algorithm Segmentation")
    print("=" * 60)
    
    K = float('inf')
    configs = treeseg_configs['icsi']
    treeseg_segments = extract_timestamp_and_text_treeseg(dataset, first_meeting, K, configs)
    
    for i, seg in enumerate(treeseg_segments):
        print(f"\n=== TreeSeg Segment {i+1} ===")
        print(f"Identifier: {seg['identifier']}")
        print(f"Time: {seg['start_time']:.2f}s - {seg['end_time']:.2f}s")
        print(f"Duration: {seg['end_time'] - seg['start_time']:.2f}s")
        print(f"Utterances: {seg['num_utterances']}")
        print(f"Text preview: {seg['text'][:150]}...")
    
    print(f"\nTotal TreeSeg segments: {len(treeseg_segments)}")
    
    write_segments_to_files(treeseg_segments, f"{first_meeting}_treeseg", "output_segments")
    
    return


if __name__ == "__main__":
    main()