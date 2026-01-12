import sys
from pathlib import Path
import importlib.util
import structlog
import os

HERE = Path(__file__).resolve()
PROJECT_DIR = HERE.parent

TREESEG_EXPLORATION = PROJECT_DIR / "treeseg_exploration"

tree_path = str(TREESEG_EXPLORATION)
sys.path.insert(0, tree_path)

configs_path = TREESEG_EXPLORATION / "configs.py"
spec = importlib.util.spec_from_file_location(
    "treeseg_exploration_configs", str(configs_path)
)

configs_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(configs_module)
treeseg_configs = configs_module.treeseg_configs

logger = structlog.get_logger(__name__)

from treeseg import TreeSeg
from utterances import extract_utterances_from_transcript_file, extract_utterances_from_transcripts
import torch

def build_lpm_config(min_segment_size, lambda_balance, context_width):
    config = {}
    config['MIN_SEGMENT_SIZE'] = min_segment_size
    config['LAMBDA_BALANCE'] = lambda_balance
    config['UTTERANCE_EXPANSION_WIDTH'] = context_width
    config['HF_EMBEDDING_MODEL'] = "sentence-transformers/all-MiniLM-L6-v2"
    config['HF_DEVICE'] = HF_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    config['HF_BATCH_SIZE'] = 32
    config['HF_NORMALIZE'] = True
    return config


def load_lpm_utterances(csv_path, data_dir, speaker, course_dir, meeting_id, max_gap_s=0.8, lowercase=True):
    if csv_path:
        logger.info("Loading utterances", source='csv', path=csv_path)
        return extract_utterances_from_transcript_file(
            csv_path=csv_path, max_gap_s=max_gap_s, lowercase=lowercase
        )

    logger.info(
        "Loading utterances",
        source='dataset',
        data_dir=data_dir,
        speaker=speaker,
        course=course_dir,
    )

    utterances = extract_utterances_from_transcripts(
        speaker=speaker,
        data_dir=data_dir,
        course_dir=course_dir,
        max_gap_s=max_gap_s,
        lowercase=lowercase,
    )

    if meeting_id:
        utterances = [utt for utt in utterances if utt.get("meeting_id") == meeting_id]
        logger.info("Filtered utterances by meeting_id", meeting_id=meeting_id, count=len(utterances))
    return utterances

# Convert the extracted utterances to treeseg compatible utterances
def build_treeseg_entries(utterances, include_ocr=False, ocr_prefix="[SLIDE] "):
    entries = []
    for idx, utt in enumerate(utterances):
        entry = dict(utt)
        entry.setdefault("utterance_index", idx)

        spoken = (entry.get("text") or "").strip()
        ocr = (entry.get("ocr_text") or "").strip()

        if include_ocr and ocr:
            composite = spoken
            if composite:
                composite = f"{composite}\n{ocr_prefix}{ocr}"
            else:
                composite = f"{ocr_prefix}{ocr}"
        else:
            composite = spoken

        entry["composite"] = composite if composite else "<blank>"
        entries.append(entry)
    return entries


def run_treeseg_on_entries(entries, target_segments):
    min_segment_size = 5
    lambda_balance = 0
    context_width = 4
    config = build_lpm_config(min_segment_size, lambda_balance, context_width)

    model = TreeSeg(configs=config, entries=list(entries))

    if target_segments is None:
        K = float('inf')
    else:
        K = target_segments
    
    transitions_hat = model.segment_meeting(K=K)


    ## this is only iterating through leaves, to get a better understanding
    ## i might have to traverse through the entire tree. 
    segments = []
    for seg_idx, leaf in enumerate(model.leaves, start=1):
        indices = leaf.segment
        segment_utts = [entries[i] for i in indices]
        start_time = segment_utts[0].get("start")
        end_time = segment_utts[-1].get("end")
        text = "\n".join(utt.get("composite", "") for utt in segment_utts).strip()

        segments.append(
            {
            "segment_id": seg_idx,
            "tree_path": leaf.identifier,
            "utterance_start": indices[0],
            "utterance_end": indices[-1],
            "n_utterances": len(indices),
            "start": start_time,
            "end": end_time,
            "text": text, 
            }
        )

    return {
        "transitions_hat": transitions_hat,
        "segments": segments, 
    }



def main():
    data_dir = os.path.join(os.path.dirname(__file__), "lpm_data")

    utterances = load_lpm_utterances(csv_path="", data_dir=data_dir, speaker='ml-1',
                                     course_dir='MultimodalMachineLearning',
                                     max_gap_s=0.8,
                                     meeting_id="",
                                     lowercase=True)
    
    entries = build_treeseg_entries(utterances, include_ocr=True)
    ret = run_treeseg_on_entries(entries, target_segments=None)

    print(ret)

if __name__ == "__main__":
    main()
        




