from re import S
import sys
import os

# pyright: reportUnreachable=false
sys.path.append(os.path.join(os.path.dirname(__file__), 'datasets'))

from datasets import ICSIDataset

def test():
    dataset = ICSIDataset(fold='dev')
    dataset.load_dataset()

    first_meeting = dataset.meetings[0]

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
    
    print("\n2. Topic Segment Analysis:")
    leaves = dataset.discover_anno_leaves(dataset.anno_roots[first_meeting])
    
    segment_sizes = [len(leaf.convo) for leaf in leaves]
    print(f"   Total segments: {len(segment_sizes)}")
    print(f"   Average segment size: {sum(segment_sizes) / len(segment_sizes):.1f} utterances")
    print(f"   Min segment size: {min(segment_sizes)}")
    print(f"   Max segment size: {max(segment_sizes)}")

def test2():
    dataset = ICSIDataset(fold='dev')
    dataset.load_dataset()

    first_meeting = dataset.meetings[0]

    root = dataset.anno_roots[first_meeting]
    
    dataset.print_anno_tree(root, first_meeting)


def main():
    dataset = ICSIDataset(fold='dev')
    dataset.load_dataset()
    meetings = dataset.meetings
    print(f"Available meetings: {meetings}")

    notes = dataset.notes[meetings[0]]
    print(f"Total notes: {len(notes)}")

    # for i, note in enumerate(notes[:5]):
        # print(f"   {i+1}. {note['composite']}")
    
    transitions = dataset.transitions[meetings[0]]
    boundary_count = sum(transitions)

    # print(f"\n   Topic boundaries: {boundary_count}")
    # print(f"   Total utterances: {len(transitions)}")

    hier_transitions = dataset.hier_transitions[meetings[0]]
    # print(f"   Hierarchical levels: {len(hier_transitions)}")


if __name__ == "__main__":
    # main()
    test2()