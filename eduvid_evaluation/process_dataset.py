from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MASTERS_THESIS_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = MASTERS_THESIS_DIR.parent
PREPROCESSING_DIR = MASTERS_THESIS_DIR / "preprocessing"

if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_DIR))

from youtube_to_lpm import (  # noqa: E402
    PipelineStepError,
    SingleVideoPipelineConfig,
    SingleVideoPipelineResult,
    run_single_video_pipeline,
)
from lpm_preprocess_lib import find_local_video_file  # noqa: E402

DEFAULT_SPLIT = "real_world_test"
DEFAULT_DATASET_PATHS = {
    "real_world_test": WORKSPACE_DIR / "eduvidqa-emnlp25" / "data" / "real_world_test.csv",
    "synthetic_test": WORKSPACE_DIR / "eduvidqa-emnlp25" / "data" / "synthetic_test.csv",
    "synthetic_train": WORKSPACE_DIR / "eduvidqa-emnlp25" / "data" / "synthetic_train.csv",
}
STORAGE_ROOT = SCRIPT_DIR / "storage"
VIDEOS_ROOT = STORAGE_ROOT / "videos"
MANIFESTS_ROOT = STORAGE_ROOT / "manifests"
PIPELINE_SETTINGS = {
    "whisper_model": "base",
    "language": "en",
    "scene_threshold": 25.0,
    "min_scene_len": 3.0,
    "text_change_filter": True,
    "text_sim_threshold": 0.85,
    "ocr_min_conf": 60.0,
    "ocr_min_tokens": 6,
    "max_same_text_span": 900.0,
    "keep_temp_audio": False,
    "delete_video_after_processing": True,
    "skip_validate": False,
    "skip_existing": True,
}


@dataclass(frozen=True)
class DatasetExample:
    example_id: str
    split: str
    row_index: int
    video_id: str
    url: str
    question: str
    answer: str
    timestamp: str
    timestamp_seconds: int


@dataclass
class VideoRecord:
    video_id: str
    url: str
    split: str
    processing_mode: str
    video_dir: Path
    question_count: int
    status: str
    reused_steps: list[str]
    error_stage: str = ""
    error_message: str = ""
    video_path: Path | None = None
    transcript_path: Path | None = None
    segments_path: Path | None = None
    slide_count: int = 0
    ocr_count: int = 0
    num_segments: int = 0
    num_words: int | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-process EduVidQA videos into per-video ASR/OCR bundles under "
            "masters-thesis/eduvid_evaluation/storage."
        )
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        choices=sorted(DEFAULT_DATASET_PATHS),
        help=f"Named EduVidQA split to process (default: {DEFAULT_SPLIT}).",
    )
    parser.add_argument(
        "--csv-path",
        help="Optional explicit CSV path. Overrides --split path selection.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N unique videos after filtering.",
    )
    parser.add_argument(
        "--video-id",
        help="Process only a single YouTube video id from the dataset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned work without writing outputs or downloading videos.",
    )
    parser.add_argument(
        "--asr",
        action="store_true",
        help="Process ASR only. Skip segmentation, slide extraction, and OCR.",
    )
    return parser


def resolve_dataset_path(split: str, csv_path: str | None) -> Path:
    if csv_path:
        return Path(csv_path).expanduser().resolve()
    return DEFAULT_DATASET_PATHS[split]


def path_for_manifest(path: Path | None) -> str:
    if path is None:
        return ""
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(WORKSPACE_DIR))
    except ValueError:
        return str(resolved)


def parse_timestamp_to_seconds(raw_timestamp: str) -> int:
    text = raw_timestamp.strip()
    parts = text.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"Unsupported timestamp format: {raw_timestamp!r}")
    try:
        numbers = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"Unsupported timestamp format: {raw_timestamp!r}") from exc
    if len(numbers) == 2:
        minutes, seconds = numbers
        return minutes * 60 + seconds
    hours, minutes, seconds = numbers
    return hours * 3600 + minutes * 60 + seconds


def load_examples(csv_path: Path, split: str) -> list[DatasetExample]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        examples = []
        for row_index, row in enumerate(reader, start=1):
            url = (row.get("url") or "").strip()
            video_id = (row.get("id") or "").strip()
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            timestamp = (row.get("timestamp") or "").strip()
            if not url or not video_id:
                raise ValueError(f"Row {row_index} is missing required url/id columns.")
            examples.append(
                DatasetExample(
                    example_id=f"{split}-{row_index:06d}",
                    split=split,
                    row_index=row_index,
                    video_id=video_id,
                    url=url,
                    question=question,
                    answer=answer,
                    timestamp=timestamp,
                    timestamp_seconds=parse_timestamp_to_seconds(timestamp),
                )
            )
    return examples


def group_examples_by_video(examples: list[DatasetExample]) -> list[tuple[str, list[DatasetExample]]]:
    grouped: dict[str, list[DatasetExample]] = defaultdict(list)
    order: list[str] = []
    for example in examples:
        if example.video_id not in grouped:
            order.append(example.video_id)
        grouped[example.video_id].append(example)
    return [(video_id, grouped[video_id]) for video_id in order]


def count_paths(video_dir: Path, pattern: str) -> int:
    return len(sorted(video_dir.glob(pattern)))


def ensure_dir(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object, dry_run: bool) -> None:
    if dry_run:
        return
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]], dry_run: bool) -> None:
    if dry_run:
        return
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str], dry_run: bool) -> None:
    if dry_run:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def questions_payload(examples: list[DatasetExample]) -> list[dict[str, object]]:
    return [
        {
            "example_id": example.example_id,
            "split": example.split,
            "row_index": example.row_index,
            "video_id": example.video_id,
            "url": example.url,
            "timestamp": example.timestamp,
            "timestamp_seconds": example.timestamp_seconds,
            "question": example.question,
            "answer": example.answer,
        }
        for example in examples
    ]


def merge_question_rows(
    existing_rows: list[dict[str, object]],
    new_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for row in existing_rows + new_rows:
        example_id = str(row["example_id"])
        merged[example_id] = row
    return [merged[example_id] for example_id in sorted(merged)]


def resolve_processing_mode(asr_only: bool) -> str:
    return "asr_only" if asr_only else "full"


def existing_artifact_path(path: Path) -> Path | None:
    return path if path.exists() else None


def build_video_metadata(
    record: VideoRecord,
    stored_question_rows: list[dict[str, object]],
) -> dict[str, object]:
    source_splits = sorted({str(row["split"]) for row in stored_question_rows})
    return {
        "video_id": record.video_id,
        "url": record.url,
        "current_split": record.split,
        "processing_mode": record.processing_mode,
        "status": record.status,
        "current_split_question_count": record.question_count,
        "stored_question_count": len(stored_question_rows),
        "source_splits": source_splits,
        "example_ids": [str(row["example_id"]) for row in stored_question_rows],
        "reused_steps": record.reused_steps,
        "error_stage": record.error_stage,
        "error_message": record.error_message,
        "artifacts": {
            "video_dir": path_for_manifest(record.video_dir),
            "video_path": path_for_manifest(record.video_path),
            "transcript_path": path_for_manifest(record.transcript_path),
            "segments_path": path_for_manifest(record.segments_path),
            "slide_count": record.slide_count,
            "ocr_count": record.ocr_count,
            "num_segments": record.num_segments,
            "num_words": record.num_words,
        },
        "processing": {
            "command": "python masters-thesis/eduvid_evaluation/process_dataset.py",
            "processing_mode": record.processing_mode,
            "asr_only": record.processing_mode == "asr_only",
            **PIPELINE_SETTINGS,
        },
    }


def process_video(
    split: str,
    video_id: str,
    examples: list[DatasetExample],
    asr_only: bool,
    dry_run: bool,
) -> VideoRecord:
    video_dir = VIDEOS_ROOT / video_id
    ensure_dir(video_dir, dry_run=dry_run)
    questions_path = video_dir / "questions.jsonl"
    question_rows = questions_payload(examples)
    existing_question_rows = [] if dry_run else read_jsonl(questions_path)
    stored_question_rows = merge_question_rows(existing_question_rows, question_rows)
    write_jsonl(questions_path, stored_question_rows, dry_run=dry_run)

    record = VideoRecord(
        video_id=video_id,
        url=examples[0].url,
        split=split,
        processing_mode=resolve_processing_mode(asr_only),
        video_dir=video_dir,
        question_count=len(examples),
        status="pending",
        reused_steps=[],
    )

    try:
        result = run_single_video_pipeline(
            SingleVideoPipelineConfig(
                youtube_url=examples[0].url,
                output_dir=video_dir,
                video_id=video_id,
                asr_only=asr_only,
                dry_run=dry_run,
                **PIPELINE_SETTINGS,
            )
        )
    except PipelineStepError as exc:
        record.status = "failed"
        record.error_stage = exc.stage
        record.error_message = str(exc)
        record.slide_count = 0 if asr_only else count_paths(video_dir, "slide_*.jpg")
        record.ocr_count = 0 if asr_only else count_paths(video_dir, "slide_*_ocr.csv")
        record.video_path = find_local_video_file(video_dir, video_id)
        record.transcript_path = existing_artifact_path(video_dir / f"{video_id}_transcripts.csv")
        record.segments_path = None if asr_only else existing_artifact_path(video_dir / "segments.txt")
        write_json(
            video_dir / "metadata.json",
            build_video_metadata(record, stored_question_rows),
            dry_run=dry_run,
        )
        return record

    record = finalize_record_from_result(record, result, dry_run=dry_run)
    write_json(
        video_dir / "metadata.json",
        build_video_metadata(record, stored_question_rows),
        dry_run=dry_run,
    )
    return record


def finalize_record_from_result(
    record: VideoRecord,
    result: SingleVideoPipelineResult,
    dry_run: bool,
) -> VideoRecord:
    record.reused_steps = list(result.reused_steps)
    record.video_path = result.video_path
    record.transcript_path = result.transcript_path
    record.segments_path = result.segments_path
    record.slide_count = len(result.slide_paths)
    record.ocr_count = len(result.ocr_paths)
    record.num_segments = result.num_segments
    record.num_words = result.num_words

    if record.processing_mode == "asr_only":
        record.segments_path = None
        record.slide_count = 0
        record.ocr_count = 0
        record.num_segments = 0

    if dry_run:
        record.status = "dry_run"
        return record

    if result.validation_result is not None and not result.validation_result.ok:
        record.status = "failed"
        record.error_stage = "validate"
        record.error_message = "; ".join(result.validation_result.errors)
        return record

    major_steps = {"transcript"} if record.processing_mode == "asr_only" else {"transcript", "segments", "slides", "ocr"}
    if major_steps.issubset(set(result.reused_steps)):
        record.status = "skipped_existing"
    else:
        record.status = "success"
    return record


def build_manifest_rows(
    examples: list[DatasetExample],
    records_by_video: dict[str, VideoRecord],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    question_rows: list[dict[str, object]] = []
    video_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []

    for record in records_by_video.values():
        video_rows.append(
            {
                "video_id": record.video_id,
                "url": record.url,
                "split": record.split,
                "processing_mode": record.processing_mode,
                "video_dir": path_for_manifest(record.video_dir),
                "question_count": record.question_count,
                "status": record.status,
                "reused_steps": ",".join(record.reused_steps),
                "video_path": path_for_manifest(record.video_path),
                "transcript_path": path_for_manifest(record.transcript_path),
                "segments_path": path_for_manifest(record.segments_path),
                "slide_count": record.slide_count,
                "ocr_count": record.ocr_count,
                "num_segments": record.num_segments,
                "num_words": record.num_words if record.num_words is not None else "",
                "error_stage": record.error_stage,
                "error_message": record.error_message,
            }
        )
        if record.status == "failed":
            failure_rows.append(
                {
                    "video_id": record.video_id,
                    "stage": record.error_stage,
                    "error_message": record.error_message,
                }
            )

    for example in examples:
        record = records_by_video[example.video_id]
        question_rows.append(
            {
                "example_id": example.example_id,
                "split": example.split,
                "row_index": example.row_index,
                "video_id": example.video_id,
                "processing_mode": record.processing_mode,
                "url": example.url,
                "timestamp": example.timestamp,
                "timestamp_seconds": example.timestamp_seconds,
                "question": example.question,
                "answer": example.answer,
                "status": record.status,
                "video_dir": path_for_manifest(record.video_dir),
                "transcript_path": path_for_manifest(record.transcript_path),
                "segments_path": path_for_manifest(record.segments_path),
            }
        )

    return question_rows, video_rows, failure_rows


def write_manifests(
    split: str,
    dataset_path: Path,
    examples: list[DatasetExample],
    records_by_video: dict[str, VideoRecord],
    processing_mode: str,
    dry_run: bool,
) -> None:
    manifest_dir = MANIFESTS_ROOT / split
    ensure_dir(manifest_dir, dry_run=dry_run)
    question_rows, video_rows, failure_rows = build_manifest_rows(examples, records_by_video)

    write_csv(
        manifest_dir / "questions.csv",
        question_rows,
        [
            "example_id",
            "split",
            "row_index",
            "video_id",
            "processing_mode",
            "url",
            "timestamp",
            "timestamp_seconds",
            "question",
            "answer",
            "status",
            "video_dir",
            "transcript_path",
            "segments_path",
        ],
        dry_run=dry_run,
    )
    write_csv(
        manifest_dir / "videos.csv",
        video_rows,
        [
            "video_id",
            "url",
            "split",
            "processing_mode",
            "video_dir",
            "question_count",
            "status",
            "reused_steps",
            "video_path",
            "transcript_path",
            "segments_path",
            "slide_count",
            "ocr_count",
            "num_segments",
            "num_words",
            "error_stage",
            "error_message",
        ],
        dry_run=dry_run,
    )
    write_csv(
        manifest_dir / "failures.csv",
        failure_rows,
        ["video_id", "stage", "error_message"],
        dry_run=dry_run,
    )

    summary = {
        "split": split,
        "processing_mode": processing_mode,
        "dataset_path": path_for_manifest(dataset_path),
        "row_count": len(examples),
        "unique_video_count": len(records_by_video),
        "success_count": sum(1 for record in records_by_video.values() if record.status == "success"),
        "skipped_existing_count": sum(
            1 for record in records_by_video.values() if record.status == "skipped_existing"
        ),
        "failure_count": sum(1 for record in records_by_video.values() if record.status == "failed"),
        "dry_run_count": sum(1 for record in records_by_video.values() if record.status == "dry_run"),
    }
    write_json(manifest_dir / "run_summary.json", summary, dry_run=dry_run)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    processing_mode = resolve_processing_mode(args.asr)

    dataset_path = resolve_dataset_path(args.split, args.csv_path)
    if not dataset_path.exists():
        parser.error(f"Dataset CSV does not exist: {dataset_path}")

    examples = load_examples(dataset_path, args.split)
    if args.video_id:
        examples = [example for example in examples if example.video_id == args.video_id]
    grouped = group_examples_by_video(examples)
    if args.limit is not None:
        grouped = grouped[: args.limit]
        allowed_video_ids = {video_id for video_id, _ in grouped}
        examples = [example for example in examples if example.video_id in allowed_video_ids]

    if not grouped:
        print("No matching videos found for the requested dataset filter.")
        return 1

    ensure_dir(VIDEOS_ROOT, dry_run=args.dry_run)
    ensure_dir(MANIFESTS_ROOT / args.split, dry_run=args.dry_run)

    records_by_video: dict[str, VideoRecord] = {}
    for idx, (video_id, video_examples) in enumerate(grouped, start=1):
        print(f"[{idx}/{len(grouped)}] Processing {video_id} ({len(video_examples)} questions)")
        records_by_video[video_id] = process_video(
            split=args.split,
            video_id=video_id,
            examples=video_examples,
            asr_only=args.asr,
            dry_run=args.dry_run,
        )

    write_manifests(
        args.split,
        dataset_path,
        examples,
        records_by_video,
        processing_mode,
        dry_run=args.dry_run,
    )

    failure_count = sum(1 for record in records_by_video.values() if record.status == "failed")
    skipped_existing_count = sum(
        1 for record in records_by_video.values() if record.status == "skipped_existing"
    )
    success_count = sum(1 for record in records_by_video.values() if record.status == "success")
    dry_run_count = sum(1 for record in records_by_video.values() if record.status == "dry_run")

    print("Run complete.")
    print(f"Dataset: {dataset_path}")
    print(f"Mode: {processing_mode}")
    print(f"Rows: {len(examples)}")
    print(f"Unique videos: {len(grouped)}")
    print(f"Successes: {success_count}")
    print(f"Skipped existing: {skipped_existing_count}")
    print(f"Failures: {failure_count}")
    if dry_run_count:
        print(f"Dry-run videos: {dry_run_count}")
    print(f"Storage root: {STORAGE_ROOT}")
    if failure_count:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
