import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import structlog
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
import ollama

from utterances import extract_utterances_from_transcript_file

HERE = Path(__file__).resolve()
PROJECT_DIR = HERE.parent

TREESEG_EXPLORATION = PROJECT_DIR / "treeseg_exploration"
sys.path.insert(0, str(TREESEG_EXPLORATION))

from treeseg import TreeSeg

logger = structlog.get_logger(__name__)
SLIDE_TOKEN = "[SLIDE]"


@dataclass(frozen=True)
class LectureDescriptor:
    speaker: str
    course_dir: str
    meeting_id: str
    video_id: str
    transcripts_path: str
    meeting_dir: str

    @property
    def key(self):
        return f"{self.speaker}/{self.course_dir}/{self.meeting_id}"

    @property
    def label(self):
        return f"{self.key} (video_id={self.video_id})"


def resolve_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_lpm_config(
    min_segment_size=5,
    lambda_balance=0,
    context_width=4,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    device=None,
    batch_size=32,
    normalize=True,
):
    if device is None:
        device = resolve_device()
    return {
        "MIN_SEGMENT_SIZE": min_segment_size,
        "LAMBDA_BALANCE": lambda_balance,
        "UTTERANCE_EXPANSION_WIDTH": context_width,
        "HF_EMBEDDING_MODEL": embedding_model,
        "HF_DEVICE": device,
        "HF_BATCH_SIZE": batch_size,
        "HF_NORMALIZE": normalize,
    }


def discover_lectures(data_dir, speaker=None, course_dir=None, meeting_ids=None):
    base = Path(data_dir)
    if speaker:
        speakers = [speaker]
    else:
        speakers = [d.name for d in base.iterdir() if d.is_dir()]

    if meeting_ids is None:
        meeting_ids = []
    elif isinstance(meeting_ids, str):
        meeting_ids = [meeting_ids]

    lectures = []
    for sp in sorted(speakers):
        sp_dir = base / sp
        if not sp_dir.is_dir():
            continue
        if course_dir:
            courses = [course_dir]
        else:
            courses = [d.name for d in sp_dir.iterdir() if d.is_dir()]
        for course in sorted(courses):
            course_path = sp_dir / course
            if not course_path.is_dir():
                continue
            if meeting_ids:
                meetings = meeting_ids
            else:
                meetings = [d.name for d in course_path.iterdir() if d.is_dir()]
            for meeting_id in sorted(meetings):
                meeting_dir = course_path / meeting_id
                if not meeting_dir.is_dir():
                    continue
                transcripts_path = None
                for fn in meeting_dir.iterdir():
                    if fn.name.endswith("_transcripts.csv"):
                        transcripts_path = fn
                        break
                if transcripts_path is None:
                    continue
                video_id = transcripts_path.name.replace("_transcripts.csv", "")
                lectures.append(
                    LectureDescriptor(
                        speaker=sp,
                        course_dir=course,
                        meeting_id=meeting_id,
                        video_id=video_id,
                        transcripts_path=str(transcripts_path),
                        meeting_dir=str(meeting_dir),
                    )
                )
    return lectures


def load_lecture_utterances(
    lecture,
    max_gap_s=0.8,
    lowercase=True,
    attach_ocr=True,
    ocr_min_conf=60.0,
    ocr_per_slide=1,
):
    segments_path = None
    slides_dir = None
    if attach_ocr:
        segments_path = os.path.join(lecture.meeting_dir, "segments.txt")
        slides_dir = lecture.meeting_dir

    utterances = extract_utterances_from_transcript_file(
        csv_path=lecture.transcripts_path,
        max_gap_s=max_gap_s,
        lowercase=lowercase,
        segments_path=segments_path,
        slides_dir=slides_dir,
        ocr_min_conf=ocr_min_conf,
        ocr_per_slide=ocr_per_slide,
    )

    for i, utt in enumerate(utterances):
        utt.update(
            {
                "meeting_id": lecture.meeting_id,
                "video_id": lecture.video_id,
                "speaker": lecture.speaker,
                "course_dir": lecture.course_dir,
                "idx": i,
                "source": "transcripts",
                "path": lecture.transcripts_path,
            }
        )
    return utterances


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


def build_segments_for_lecture(
    lecture,
    utterances,
    treeseg_config,
    target_segments=None,
    include_ocr=True,
    ocr_prefix="[SLIDE] ",
):
    entries = build_treeseg_entries(
        utterances, include_ocr=include_ocr, ocr_prefix=ocr_prefix
    )
    if not entries:
        return []

    model = TreeSeg(configs=treeseg_config, entries=list(entries))
    k = float("inf") if target_segments is None else target_segments
    model.segment_meeting(K=k)

    segments = []
    for seg_idx, leaf in enumerate(model.leaves, start=1):
        indices = leaf.segment
        segment_utts = [entries[i] for i in indices]
        start_time = segment_utts[0].get("start")
        end_time = segment_utts[-1].get("end")
        text = "\n".join(utt.get("composite", "") for utt in segment_utts).strip()
        if not text:
            text = "<blank>"

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
                "lecture_key": lecture.key,
                "speaker": lecture.speaker,
                "course_dir": lecture.course_dir,
                "meeting_id": lecture.meeting_id,
                "video_id": lecture.video_id,
            }
        )
    return segments


def split_segment_text(text, slide_token=SLIDE_TOKEN):
    if not text:
        return "", ""
    spoken_lines = []
    ocr_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(slide_token):
            ocr_line = stripped[len(slide_token) :].lstrip(" :\t")
            if ocr_line:
                ocr_lines.append(ocr_line)
        else:
            spoken_lines.append(stripped)
    return "\n".join(spoken_lines).strip(), "\n".join(ocr_lines).strip()


def build_rerank_input(text, slide_token=SLIDE_TOKEN):
    spoken, ocr = split_segment_text(text, slide_token=slide_token)
    segment_block = spoken if spoken else "<blank>"
    parts = [f"Segment:\n{segment_block}"]
    if ocr:
        ocr_lines = [line.strip() for line in ocr.splitlines() if line.strip()]
        ocr_block = "\n".join(f"{slide_token} {line}" for line in ocr_lines)
        parts.append(f"{slide_token}\nSlide OCR:\n{ocr_block}")
    else:
        parts.append("Slide OCR:\n<blank>")
    return "\n\n".join(parts)


class CrossEncoderReranker:
    def __init__(self, model_name, device=None):
        if device is None:
            device = resolve_device()
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query, results, top_n=5, slide_token=SLIDE_TOKEN):
        if not results:
            return []
        pairs = []
        for hit in results:
            rerank_text = build_rerank_input(hit.get("text", ""), slide_token=slide_token)
            pairs.append((query, rerank_text))
        scores = self.model.predict(pairs)
        rescored = []
        for score, hit in zip(scores, results):
            updated = dict(hit)
            updated["rerank_score"] = float(score)
            rescored.append(updated)
        rescored.sort(key=lambda item: item["rerank_score"], reverse=True)
        if top_n is None:
            return rescored
        return rescored[: min(top_n, len(rescored))]


class LpmVectorIndex:
    def __init__(self, model_name, device=None, normalize=True, build_global=True):
        if device is None:
            device = resolve_device()
        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize
        self.build_global = build_global
        self.lecture_indices = {}
        self.global_segments = []
        self.global_index = None
        self._global_embeddings = []

    def _build_faiss_index(self, embeddings):
        dim = embeddings.shape[1]
        if self.normalize:
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index

    def add_lecture(self, lecture, segments):
        if not segments:
            return
        texts = [s["text"] for s in segments]
        embeddings = self.model.encode(texts, normalize_embeddings=self.normalize)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        index = self._build_faiss_index(embeddings)

        self.lecture_indices[lecture.key] = {
            "lecture": lecture,
            "segments": segments,
            "index": index,
        }

        if self.build_global:
            self._global_embeddings.append(embeddings)
            self.global_segments.extend(segments)

    def finalize(self):
        if not self.build_global or not self._global_embeddings:
            return
        all_embeddings = np.vstack(self._global_embeddings).astype(np.float32)
        self.global_index = self._build_faiss_index(all_embeddings)
        self._global_embeddings = []

    def search(self, query, top_k=5, lecture_key=None):
        if lecture_key:
            entry = self.lecture_indices.get(lecture_key)
            if entry is None:
                raise KeyError(f"Unknown lecture key: {lecture_key}")
            index = entry["index"]
            segments = entry["segments"]
        else:
            index = self.global_index
            segments = self.global_segments
        if index is None:
            raise ValueError("Index not built yet.")
        if not segments:
            return []
        k = min(top_k, len(segments))

        q = self.model.encode([query], normalize_embeddings=self.normalize)
        q = np.asarray(q, dtype=np.float32)
        scores, idxs = index.search(q, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            seg = segments[idx]
            results.append(
                {
                    "score": float(score),
                    "lecture_key": seg.get("lecture_key"),
                    "speaker": seg.get("speaker"),
                    "course_dir": seg.get("course_dir"),
                    "meeting_id": seg.get("meeting_id"),
                    "video_id": seg.get("video_id"),
                    "segment_id": seg.get("segment_id"),
                    "tree_path": seg.get("tree_path"),
                    "utterance_start": seg.get("utterance_start"),
                    "utterance_end": seg.get("utterance_end"),
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": seg.get("text"),
                }
            )
        return results


def format_lecture_list(lectures):
    lines = []
    for idx, lecture in enumerate(lectures, start=1):
        lines.append(f"{idx:02d}. {lecture.label}")
    return "\n".join(lines)


def resolve_lecture_choice(lectures, choice, allow_all=True):
    if not choice:
        return None
    value = choice.strip()
    if not value:
        return None
    lowered = value.lower()
    if lowered in {"all", "a"}:
        if allow_all:
            return None
        raise ValueError("Global search is disabled. Pick a lecture.")
    if value.isdigit():
        idx = int(value) - 1
        if 0 <= idx < len(lectures):
            return lectures[idx].key
        raise ValueError("Lecture index out of range.")
    exact = [lec for lec in lectures if lec.key == value]
    if exact:
        return exact[0].key
    meeting_matches = [lec for lec in lectures if lec.meeting_id == value]
    if len(meeting_matches) == 1:
        return meeting_matches[0].key
    if meeting_matches:
        raise ValueError("Meeting id matches multiple lectures. Use full key.")
    raise ValueError("Unknown lecture choice.")


def print_results(results, max_chars=500):
    for rank, hit in enumerate(results, start=1):
        if "rerank_score" in hit:
            header = (
                f"{rank}. {hit['rerank_score']:.3f} (rerank) | "
                f"{hit['lecture_key']} | seg {hit['segment_id']} "
                f"({hit['start']}-{hit['end']}s)"
            )
        else:
            header = (
                f"{rank}. {hit['score']:.3f} | {hit['lecture_key']} | seg "
                f"{hit['segment_id']} ({hit['start']}-{hit['end']}s)"
            )
        print(header)
        text = hit["text"] or ""
        # if max_chars and len(text) > max_chars:
        #     text = text[:max_chars].rstrip() + "..."
        if text:
            for line in text.splitlines():
                print(f"    {line}")
        print()


def _format_context_header(hit, rank):
    parts = [f"[{rank}]"]
    score = hit.get("rerank_score", hit.get("score"))
    if isinstance(score, (int, float)):
        parts.append(f"score={score:.3f}")
    lecture_key = hit.get("lecture_key")
    if lecture_key:
        parts.append(str(lecture_key))
    segment_id = hit.get("segment_id")
    if segment_id is not None:
        parts.append(f"seg={segment_id}")
    start = hit.get("start")
    end = hit.get("end")
    if start is not None and end is not None:
        parts.append(f"time={start}-{end}s")
    return " ".join(parts)


def build_context(results, max_chars=8000, include_ocr=True, slide_token=SLIDE_TOKEN):
    if not results:
        return ""

    blocks = []
    total_chars = 0
    for rank, hit in enumerate(results, start=1):
        text = (hit.get("text") or "").strip()
        if not text:
            continue

        spoken, ocr = split_segment_text(text, slide_token=slide_token)
        header = _format_context_header(hit, rank)
        parts = [header, f"Spoken:\n{spoken if spoken else '<blank>'}"]
        if include_ocr:
            parts.append(f"Slide OCR:\n{ocr if ocr else '<blank>'}")
        block = "\n".join(parts)

        if max_chars is not None and total_chars + len(block) > max_chars:
            break

        blocks.append(block)
        total_chars += len(block) + 2

    return "\n\n".join(blocks).strip()


def query_response(
    query,
    context,
    model="llama3.2",
    system_prompt=None,
    temperature=0.2,
    keep_alive=None,
    client=None,
    host=None,
):
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    if context is None:
        context = ""

    if system_prompt is None:
        system_prompt = (
            "Based on the following lecture transcript and slide segments, answer the question to the best of your abilities." \
            "Utilize ONLY the below context as your reference for generating the answer. It is fine if the answer is not DIRECTLY stated" \
            "but if you can infer the answer from the text return that answer. "
        )

    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    options = {"temperature": temperature} if temperature is not None else None

    if client is None:
        if host:
            client = ollama.Client(host=host)
        else:
            client = ollama

    response = client.chat(
        model=model,
        messages=messages,
        options=options,
        keep_alive=keep_alive,
    )

    message = getattr(response, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if content is not None:
            return str(content).strip()

    if isinstance(response, dict):
        message = response.get("message") or {}
        content = message.get("content")
        if content is not None:
            return str(content).strip()
        response_text = response.get("response")
        if response_text is not None:
            return str(response_text).strip()

    return str(response).strip()


def build_vector_store(
    lectures,
    treeseg_config,
    embed_model="sentence-transformers/all-MiniLM-L6-v2",
    device=None,
    normalize=True,
    build_global=True,
    max_gap_s=0.8,
    lowercase=True,
    attach_ocr=True,
    ocr_min_conf=60.0,
    ocr_per_slide=1,
    target_segments=None,
):
    store = LpmVectorIndex(
        model_name=embed_model,
        device=device,
        normalize=normalize,
        build_global=build_global,
    )

    for lecture in lectures:
        logger.info("Indexing lecture", lecture=lecture.key)
        utterances = load_lecture_utterances(
            lecture,
            max_gap_s=max_gap_s,
            lowercase=lowercase,
            attach_ocr=attach_ocr,
            ocr_min_conf=ocr_min_conf,
            ocr_per_slide=ocr_per_slide,
        )
        if not utterances:
            logger.warning("No utterances found", lecture=lecture.key)
            continue
        segments = build_segments_for_lecture(
            lecture,
            utterances,
            treeseg_config=treeseg_config,
            target_segments=target_segments,
            include_ocr=attach_ocr,
        )
        if not segments:
            logger.warning("No segments built", lecture=lecture.key)
            continue
        store.add_lecture(lecture, segments)

    store.finalize()
    return store


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a TreeSeg-based vector index over LPM lectures."
    )
    parser.add_argument(
        "--list-lectures",
        action="store_true",
        help="List available lectures and exit.",
    )
    parser.add_argument(
        "--lecture",
        default=None,
        help="Lecture key or number (use with --list-lectures to see options).",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Single query to run (non-interactive).",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Rerank retrieved segments with a cross-encoder.",
    )
    parser.add_argument(
        "--rerank-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model for reranking.",
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=5,
        help="Number of results to return after reranking.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = PROJECT_DIR / "lpm_data"
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    top_k = 50
    top_n = args.rerank_top_n
    max_chars = 500

    lectures = discover_lectures(
        data_dir=data_dir,
    )
    if not lectures:
        print("No lectures found. Check the data directory.")
        return

    if args.list_lectures:
        print(format_lecture_list(lectures))
        if args.query is None and args.lecture is None:
            return

    treeseg_config = build_lpm_config(
        embedding_model=embedding_model,
    )

    if args.lecture:
        try:
            lecture_key = resolve_lecture_choice(lectures, args.lecture, allow_all=False)
        except ValueError as exc:
            print(str(exc))
            return
        target_lectures = [lec for lec in lectures if lec.key == lecture_key]
    else:
        print("Available lectures:")
        print(format_lecture_list(lectures))
        print("Choose a lecture by number or key.")
        while True:
            choice = input("lecture> ").strip()
            if not choice:
                continue
            try:
                lecture_key = resolve_lecture_choice(
                    lectures, choice, allow_all=False
                )
                break
            except ValueError as exc:
                print(str(exc))
                continue
        target_lectures = [lec for lec in lectures if lec.key == lecture_key]

    store = build_vector_store(
        target_lectures,
        treeseg_config=treeseg_config,
        embed_model=embedding_model,
        normalize=True,
        build_global=False,
        max_gap_s=0.8,
        lowercase=True,
        attach_ocr=True,
        ocr_min_conf=60.0,
        ocr_per_slide=1,
        target_segments=None,
    )

    reranker = None
    if args.rerank:
        reranker = CrossEncoderReranker(args.rerank_model, device=resolve_device())

    if args.query:
        if not args.query.strip():
            print("Empty query. Provide text for --query.")
            return
        results = store.search(args.query, top_k=top_k, lecture_key=lecture_key)
        if reranker:
            results = reranker.rerank(args.query, results, top_n=top_n)
        print_results(results, max_chars=max_chars)
        return

    print("Type a query to search (empty or 'exit' to quit).")
    while True:
        query = input("search> ").strip()
        if not query or query.lower() in {"exit", "quit", "q"}:
            break
        results = store.search(query, top_k=top_k, lecture_key=lecture_key)
        if reranker:
            results = reranker.rerank(query, results, top_n=top_n)

        context = build_context(results)
        response = query_response(query, context)
        print(response)

        



if __name__ == "__main__":
    main()
