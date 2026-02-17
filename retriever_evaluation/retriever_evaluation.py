# %%
import ast
import json
import re
import sys
from pathlib import Path
from IPython.display import display as ipy_display
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)

HERE = Path(__file__).resolve()
RETRIEVER_EVAL_DIR = HERE.parent
PROJECT_DIR = HERE.parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from treeseg_vector_index import (
    CrossEncoderReranker,
    build_lpm_config,
    build_ocr_vector_store,
    build_rerank_input_ocr,
    build_vector_store,
    discover_lectures,
    resolve_device,
)

ASR_SEGMENTS = RETRIEVER_EVAL_DIR / "segment_dumps" / "asr_segments.json"
OCR_SLIDES = RETRIEVER_EVAL_DIR / "segment_dumps" / "ocr_slides.json"
RETRIEVER_DATASET = RETRIEVER_EVAL_DIR / "retriever_evaluation.csv"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OCR_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = 50
TOP_N = 10

TARGET_SPEAKER = "anat-1"
TARGET_COURSE = "AnatomyPhysiology"
TARGET_MEETING_IDS = ["01"]

data_dir = PROJECT_DIR / "lpm_data"
target_lectures = discover_lectures(
    data_dir=data_dir,
    speaker=TARGET_SPEAKER,
    course_dir=TARGET_COURSE,
    meeting_ids=TARGET_MEETING_IDS,
)


def parse_id_list(cell):
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    text = str(cell).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [int(v) for v in parsed]
        if isinstance(parsed, int):
            return [int(parsed)]
    except (ValueError, SyntaxError):
        pass
    return [int(v) for v in re.findall(r"\d+", text)]


def build_asr_lookup(asr_json):
    lookup = {}
    rows = asr_json if isinstance(asr_json, list) else [asr_json]
    for row in rows:
        lecture_key = row.get("lecture_key")
        if not lecture_key:
            continue
        segment_map = {}
        for segment in row.get("asr_segments", []):
            segment_id = segment.get("segment_id")
            if segment_id is None:
                continue
            segment_map[int(segment_id)] = segment.get("text", "")
        lookup[lecture_key] = segment_map
    return lookup


def build_ocr_lookup(ocr_json):
    lookup = {}
    rows = ocr_json if isinstance(ocr_json, list) else [ocr_json]
    for row in rows:
        lecture_key = row.get("lecture_key")
        if not lecture_key:
            continue
        slide_map = {}
        for slide in row.get("ocr_slides", []):
            slide_index = slide.get("slide_index")
            if slide_index is None:
                continue
            slide_map[int(slide_index)] = slide.get("text", "")
        lookup[lecture_key] = slide_map
    return lookup


def compute_id_metrics(gt_ids, pred_ids):
    gt_set = set(int(v) for v in gt_ids)
    pred_ids = [int(v) for v in pred_ids]
    if not gt_set:
        return {"hit": 0.0, "recall": 0.0, "mrr": 0.0}

    pred_set = set(pred_ids)
    overlap = gt_set.intersection(pred_set)
    hit = 1.0 if overlap else 0.0
    recall = len(overlap) / float(len(gt_set))

    mrr = 0.0
    for idx, pred_id in enumerate(pred_ids, start=1):
        if pred_id in gt_set:
            mrr = 1.0 / float(idx)
            break
    return {"hit": hit, "recall": recall, "mrr": mrr}


def evaluate_combined_row(
    row,
    combined_store,
    combined_reranker,
    asr_lookup,
    ocr_lookup,
    top_k,
    top_n,
):
    lecture_key = str(row.lecture_key).strip()
    question = row.question
    
    gt_asr_ids = parse_id_list(row.asr_segment_ids)
    gt_ocr_ids = parse_id_list(row.ocr_slide_indices)

    asr_map = asr_lookup[lecture_key]
    ocr_map = ocr_lookup[lecture_key]
    gt_asr_ids = [sid for sid in gt_asr_ids if sid in asr_map]
    gt_ocr_ids = [idx for idx in gt_ocr_ids if idx in ocr_map]

    results = combined_store.search(question, top_k=top_k, lecture_key=lecture_key)
    results = combined_reranker.rerank(question, results, top_n=top_n)

    pred_asr_ids = [
        int(hit["segment_id"])
        for hit in results
        if hit.get("segment_id") is not None
    ]
    pred_text = "\n".join(hit.get("text", "") for hit in results if hit.get("text"))
    gt_ocr_text = "\n".join(ocr_map[idx] for idx in gt_ocr_ids if ocr_map.get(idx))

    combined_metrics = compute_id_metrics(gt_asr_ids, pred_asr_ids)

    return {
        "question": question,
        "lecture_key": lecture_key,
        "combined_hit@N": combined_metrics["hit"],
        "combined_recall@N": combined_metrics["recall"],
        "combined_mrr": combined_metrics["mrr"],
        "gt_asr_ids": gt_asr_ids,
        "pred_asr_ids": pred_asr_ids,
        "gt_ocr_indices": gt_ocr_ids,
    }


def evaluate_separate_row(
    row,
    asr_store,
    ocr_store,
    asr_reranker,
    ocr_reranker,
    asr_lookup,
    ocr_lookup,
    top_k,
    top_n,
):
    lecture_key = str(row.lecture_key).strip()
    question = row.question

    gt_asr_ids = parse_id_list(row.asr_segment_ids)
    gt_ocr_ids = parse_id_list(row.ocr_slide_indices)

    asr_map = asr_lookup[lecture_key]
    ocr_map = ocr_lookup[lecture_key]
    gt_asr_ids = [sid for sid in gt_asr_ids if sid in asr_map]
    gt_ocr_ids = [idx for idx in gt_ocr_ids if idx in ocr_map]

    asr_results = asr_store.search(question, top_k=top_k, lecture_key=lecture_key)
    asr_results = asr_reranker.rerank(question, asr_results, top_n=top_n)
    pred_asr_ids = [
        int(hit["segment_id"])
        for hit in asr_results
        if hit.get("segment_id") is not None
    ]
    asr_metrics = compute_id_metrics(gt_asr_ids, pred_asr_ids)

    ocr_results = []
    if lecture_key in ocr_store.lecture_indices:
        ocr_results = ocr_store.search(question, top_k=top_k, lecture_key=lecture_key)
        ocr_results = ocr_reranker.rerank(question, ocr_results, top_n=top_n)
    pred_ocr_ids = [
        int(hit["slide_index"])
        for hit in ocr_results
        if hit.get("slide_index") is not None
    ]
    ocr_metrics = compute_id_metrics(gt_ocr_ids, pred_ocr_ids)

    return {
        "question": question,
        "lecture_key": lecture_key,
        "asr_hit@N": asr_metrics["hit"],
        "asr_recall@N": asr_metrics["recall"],
        "asr_mrr": asr_metrics["mrr"],
        "ocr_hit@N": ocr_metrics["hit"],
        "ocr_recall@N": ocr_metrics["recall"],
        "ocr_mrr": ocr_metrics["mrr"],
        "gt_asr_ids": gt_asr_ids,
        "pred_asr_ids": pred_asr_ids,
        "gt_ocr_indices": gt_ocr_ids,
        "pred_ocr_indices": pred_ocr_ids,
    }


def print_mean_summary(df, title, metric_cols):
    print(f"\n{title}")
    if df.empty:
        print("No rows evaluated.")
        return
    print(f"rows: {len(df)}")
    for col in metric_cols:
        if col in df.columns:
            print(f"{col}: {df[col].mean():.4f}")


def display_df(df, title):
    print(f"\n{title}")
    ipy_display(df)


class RetrieverEvaluation:
    def __init__(self, eval_dataset_path):
        self.retriever_dataset = pd.read_csv(eval_dataset_path, skipinitialspace=True)

    def read_json(self, filepath):
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)

    def get_eval_df(self):
        return self.retriever_dataset

    def create_combined_retriever(self):
        treeseg_config = build_lpm_config()
        store = build_vector_store(
            target_lectures,
            treeseg_config=treeseg_config,
            embed_model=EMBEDDING_MODEL,
            normalize=True,
            build_global=False,
            max_gap_s=0.8,
            lowercase=True,
            attach_ocr=True,
            ocr_min_conf=60.0,
            ocr_per_slide=1,
            target_segments=None,
        )
        reranker = CrossEncoderReranker(RERANK_MODEL, device=resolve_device())
        return store, reranker

    def create_separate_retriever(self):
        treeseg_config = build_lpm_config()
        asr_store = build_vector_store(
            target_lectures,
            treeseg_config=treeseg_config,
            embed_model=EMBEDDING_MODEL,
            normalize=True,
            build_global=False,
            max_gap_s=0.8,
            lowercase=True,
            attach_ocr=True,
            include_ocr_in_treeseg=False,
            ocr_min_conf=60.0,
            ocr_per_slide=1,
            target_segments=None,
        )
        ocr_store = build_ocr_vector_store(
            target_lectures,
            embed_model=EMBEDDING_MODEL,
            normalize=True,
            build_global=False,
            ocr_min_conf=60.0,
        )
        asr_reranker = CrossEncoderReranker(RERANK_MODEL, device=resolve_device())
        ocr_reranker = CrossEncoderReranker(
            OCR_RERANK_MODEL,
            device=resolve_device(),
            input_builder=build_rerank_input_ocr,
        )
        return asr_store, ocr_store, asr_reranker, ocr_reranker


def main():
    evaluator = RetrieverEvaluation(RETRIEVER_DATASET)
    eval_df = evaluator.get_eval_df()
    asr_json = evaluator.read_json(ASR_SEGMENTS)
    ocr_json = evaluator.read_json(OCR_SLIDES)

    asr_lookup = build_asr_lookup(asr_json)
    ocr_lookup = build_ocr_lookup(ocr_json)

    combined_store, combined_reranker = evaluator.create_combined_retriever()
    asr_store, ocr_store, asr_reranker, ocr_reranker = evaluator.create_separate_retriever()

    combined_rows = []
    separate_rows = []

    for row in eval_df.itertuples(index=False):
        combined_result = evaluate_combined_row(
            row=row,
            combined_store=combined_store,
            combined_reranker=combined_reranker,
            asr_lookup=asr_lookup,
            ocr_lookup=ocr_lookup,
            top_k=TOP_K,
            top_n=TOP_N,
        )
        if combined_result is not None:
            combined_rows.append(combined_result)

        separate_result = evaluate_separate_row(
            row=row,
            asr_store=asr_store,
            ocr_store=ocr_store,
            asr_reranker=asr_reranker,
            ocr_reranker=ocr_reranker,
            asr_lookup=asr_lookup,
            ocr_lookup=ocr_lookup,
            top_k=TOP_K,
            top_n=TOP_N,
        )
        if separate_result is not None:
            separate_rows.append(separate_result)

    combined_results_df = pd.DataFrame(combined_rows)
    separate_results_df = pd.DataFrame(separate_rows)

    display_df(combined_results_df, "Combined Results")
    display_df(separate_results_df, "Separate Results")

    return combined_results_df, separate_results_df


if __name__ == "__main__":
    main()

# %%
