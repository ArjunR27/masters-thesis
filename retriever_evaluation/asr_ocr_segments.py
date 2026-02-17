from pathlib import Path
import json
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from utterances import extract_utterances_from_transcript_file
from treeseg_vector_index import (
    build_lpm_config,
    build_ocr_slide_entries,
    build_segments_for_lecture,
    discover_lectures,
)


# Simple knobs (no argparse)
LECTURE_CHOICE = None  # None for all, or set to a lecture key like "anat-1/AnatomyPhysiology/01"
MAX_GAP_S = 0.8
OCR_MIN_CONF = 60.0
OCR_PER_SLIDE = 1
ASR_OUT_PATH = ROOT_DIR / "retriever_evaluation" / "segment_dumps" / "asr_segments.json"
OCR_OUT_PATH = ROOT_DIR / "retriever_evaluation" / "segment_dumps" / "ocr_slides.json"


def main():
    data_dir = ROOT_DIR / "lpm_data"
    lectures = discover_lectures(data_dir=data_dir)
    if not lectures:
        print("No lectures found. Check the data directory.")
        return

    treeseg_config = build_lpm_config(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )
    asr_results = []
    ocr_results = []

    lecture = lectures[0]
    utterances = extract_utterances_from_transcript_file(
        csv_path=lecture.transcripts_path,
        max_gap_s=MAX_GAP_S,
        lowercase=True,
        segments_path=None,
        slides_dir=lecture.meeting_dir,
        ocr_min_conf=OCR_MIN_CONF,
        ocr_per_slide=OCR_PER_SLIDE,
    )

    segments = build_segments_for_lecture(
        lecture,
        utterances,
        treeseg_config=treeseg_config,
        target_segments=None,
        include_ocr=False,
    )

    slides = build_ocr_slide_entries(
        lecture, ocr_min_conf=OCR_MIN_CONF, line_sep="\n"
    )

    asr_results.append(
        {
            "lecture_key": lecture.key,
            "speaker": lecture.speaker,
            "course_dir": lecture.course_dir,
            "meeting_id": lecture.meeting_id,
            "video_id": lecture.video_id,
            "asr_segments": [
                {
                    "segment_id": seg.get("segment_id"),
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": seg.get("text"),
                }
                for seg in segments
            ],
        }
    )

    ocr_results.append(
            {
                "lecture_key": lecture.key,
                "speaker": lecture.speaker,
                "course_dir": lecture.course_dir,
                "meeting_id": lecture.meeting_id,
                "video_id": lecture.video_id,
                "ocr_slides": [
                    {k: v for k, v in slide.items() if k != "segment_id"}
                    for slide in slides
                ],
            }
        )

    ASR_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ASR_OUT_PATH.write_text(json.dumps(asr_results, indent=2), encoding="utf-8")
    OCR_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OCR_OUT_PATH.write_text(json.dumps(ocr_results, indent=2), encoding="utf-8")
    print(f"Wrote {ASR_OUT_PATH}")
    print(f"Wrote {OCR_OUT_PATH}")


if __name__ == "__main__":
    main()
    
