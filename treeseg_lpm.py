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
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


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


def load_lpm_utterances(
    csv_path,
    data_dir,
    speaker,
    course_dir,
    meeting_id,
    max_gap_s=0.8,
    lowercase=True,
    attach_ocr=True,
    ocr_min_conf=0.0,
    ocr_per_slide=1,
):
    if csv_path:
        logger.info("Loading utterances", source='csv', path=csv_path)
        transcript_dir = os.path.dirname(csv_path)
        segments_path = os.path.join(transcript_dir, "segments.txt")
        slides_dir = transcript_dir
        return extract_utterances_from_transcript_file(
            csv_path=csv_path,
            max_gap_s=max_gap_s,
            lowercase=lowercase,
            segments_path=segments_path if attach_ocr else None,
            slides_dir=slides_dir if attach_ocr else None,
            ocr_min_conf=ocr_min_conf,
            ocr_per_slide=ocr_per_slide,
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
        attach_ocr=attach_ocr,
        ocr_min_conf=ocr_min_conf,
        ocr_per_slide=ocr_per_slide,
    )

    if meeting_id:
        utterances = [utt for utt in utterances if utt.get("meeting_id") == meeting_id]
        logger.info("Filtered utterances by meeting_id", meeting_id=meeting_id, count=len(utterances))
    return utterances

# Convert the extracted utterances to treeseg compatible utterances
def build_treeseg_entries(utterances, include_ocr=True, ocr_prefix="[SLIDE] "):
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
            "is_leaf": True,
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

def demo_vector_index_retrieval(output, interactive=True, top_k=10):
    segments = [s for s in output["segments"] if s.get("is_leaf", True)]
    seg_texts = [s["text"] for s in segments]
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    seg_emb = model.encode(seg_texts, normalize_embeddings=True)
    seg_emb = np.asarray(seg_emb, dtype=np.float32)

    index = faiss.IndexFlatIP(seg_emb.shape[1])
    index.add(seg_emb)

    def search_segments(query, k=10):
        q = model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, idxs = index.search(q, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            seg = segments[idx]
            results.append({
                "score": float(score),
                "segment_id": seg["segment_id"],
                "tree_path": seg["tree_path"],
                "utterance_start": seg["utterance_start"],
                "utterance_end": seg["utterance_end"],
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": seg["text"],
            })
        
        return results

    if not interactive or not sys.stdin.isatty():
        hits = search_segments("what are the four eras of machine", k=top_k)
        for h in hits:
            print(f"{h['score']:.3f} | seg {h['segment_id']} ({h['start']}–{h['end']}s)")
            print(h['text'], "\n")
        return

    print("Type a query to search (empty or 'exit' to quit).")
    while True:
        query = input("search> ").strip()
        if not query or query.lower() in {"exit", "quit", "q"}:
            break
        hits = search_segments(query, k=top_k)
        for h in hits:
            print(f"{h['score']:.3f} | seg {h['segment_id']} ({h['start']}–{h['end']}s)")
            print(h['text'], "\n")



def main():
    data_dir = os.path.join(os.path.dirname(__file__), "lpm_data")

    utterances = load_lpm_utterances(csv_path="", data_dir=data_dir, speaker='ml-1',
                                     course_dir='MultimodalMachineLearning',
                                     max_gap_s=0.8,
                                     meeting_id="",
                                     lowercase=True,
                                     attach_ocr=True,
                                     ocr_min_conf=60.0,
                                     ocr_per_slide=1)
    
    entries = build_treeseg_entries(utterances, include_ocr=True, ocr_prefix="[SLIDE] ")
    ret = run_treeseg_on_entries(entries, target_segments=None)
    demo_vector_index_retrieval(ret)

if __name__ == "__main__":
    main()
        
