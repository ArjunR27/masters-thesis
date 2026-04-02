from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

from lpm_preprocess_lib import (
    DEFAULT_MIN_SCENE_LEN,
    DEFAULT_OCR_MASK_BOTTOM_RATIO,
    DEFAULT_OCR_MASK_LEFT_RATIO,
    DEFAULT_OCR_MASK_RIGHT_RATIO,
    DEFAULT_OCR_MASK_TOP_RATIO,
    DEFAULT_SCENE_THRESHOLD,
    DEFAULT_SLIDE_DEDUPE_HAMMING_DISTANCE,
    DEFAULT_TEXT_SIM_THRESHOLD,
    dedupe_slide_images,
    detect_scene_end_times,
    download_video,
    ensure_meeting_dir,
    extract_audio_to_wav,
    extract_slide_images,
    fetch_video_id,
    find_local_video_file,
    filter_end_times_by_ocr_text_change,
    print_validation_result,
    require_commands,
    run_ocr_on_slides,
    transcribe_with_whisper,
    TRANSCRIPT_COLUMNS,
    ValidationResult,
    validate_meeting_dir,
    write_segments_txt,
)


@dataclass(frozen=True)
class SingleVideoPipelineConfig:
    youtube_url: str
    output_dir: Path
    video_id: str | None = None
    asr_only: bool = False
    whisper_model: str = "base"
    language: str = "en"
    scene_threshold: float = DEFAULT_SCENE_THRESHOLD
    min_scene_len: float = DEFAULT_MIN_SCENE_LEN
    text_change_filter: bool = True
    text_sim_threshold: float = DEFAULT_TEXT_SIM_THRESHOLD
    ocr_min_conf: float = 60.0
    ocr_min_tokens: int = 6
    max_same_text_span: float = 900.0
    ocr_mask_top_ratio: float = DEFAULT_OCR_MASK_TOP_RATIO
    ocr_mask_bottom_ratio: float = DEFAULT_OCR_MASK_BOTTOM_RATIO
    ocr_mask_left_ratio: float = DEFAULT_OCR_MASK_LEFT_RATIO
    ocr_mask_right_ratio: float = DEFAULT_OCR_MASK_RIGHT_RATIO
    slide_dedupe_hamming_distance: int = DEFAULT_SLIDE_DEDUPE_HAMMING_DISTANCE
    keep_temp_audio: bool = False
    delete_video_after_processing: bool = False
    skip_validate: bool = False
    dry_run: bool = False
    skip_existing: bool = False


@dataclass
class SingleVideoPipelineResult:
    output_dir: Path
    video_id: str
    video_path: Path | None
    transcript_path: Path
    segments_path: Path | None
    slide_paths: list[Path] = field(default_factory=list)
    ocr_paths: list[Path] = field(default_factory=list)
    num_segments: int = 0
    num_words: int | None = None
    reused_steps: list[str] = field(default_factory=list)
    validation_result: ValidationResult | None = None


class PipelineStepError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def _transcript_is_valid(transcript_path: Path) -> bool:
    if not transcript_path.exists():
        return False
    with transcript_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return False
        if header != TRANSCRIPT_COLUMNS:
            return False
        try:
            next(reader)
        except StopIteration:
            return False
    return True


def _count_csv_rows(csv_path: Path) -> int:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        return sum(1 for _ in reader)


def _count_segments(segments_path: Path) -> int:
    with segments_path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _list_paths(output_dir: Path, pattern: str) -> list[Path]:
    return sorted(output_dir.glob(pattern))


def _validate_transcript_file(transcript_path: Path) -> ValidationResult:
    errors: list[str] = []
    if not transcript_path.exists():
        return ValidationResult(False, [f"Missing required transcript file: {transcript_path}"], [])

    with transcript_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return ValidationResult(False, [f"Transcript file is empty: {transcript_path}"], [])
        if header != TRANSCRIPT_COLUMNS:
            errors.append(
                f"Transcript header must be exactly {TRANSCRIPT_COLUMNS}, found {header}"
            )
        if next(reader, None) is None:
            errors.append(f"Transcript file has no transcript rows: {transcript_path}")
    return ValidationResult(ok=not errors, errors=errors, warnings=[])


def _build_existing_bundle_result(
    output_dir: Path,
    resolved_video_id: str,
    existing_video_path: Path | None,
    validation_result: ValidationResult | None,
    *,
    asr_only: bool = False,
) -> SingleVideoPipelineResult:
    transcript_path = output_dir / f"{resolved_video_id}_transcripts.csv"
    segments_path = None if asr_only else output_dir / "segments.txt"
    reused_steps = ["transcript"] if asr_only else ["transcript", "segments", "slides", "ocr"]
    if existing_video_path is not None:
        reused_steps.append("video")
    return SingleVideoPipelineResult(
        output_dir=output_dir,
        video_id=resolved_video_id,
        video_path=existing_video_path,
        transcript_path=transcript_path,
        segments_path=segments_path,
        slide_paths=[] if asr_only else _list_paths(output_dir, "slide_*.jpg"),
        ocr_paths=[] if asr_only else _list_paths(output_dir, "slide_*_ocr.csv"),
        num_segments=0 if asr_only else _count_segments(segments_path),
        num_words=_count_csv_rows(transcript_path),
        reused_steps=reused_steps,
        validation_result=validation_result,
    )


def _delete_file_if_present(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    path.unlink()


def _clear_segment_artifacts(output_dir: Path) -> None:
    for pattern in ("segments.txt", "slide_*.jpg", "slide_*_ocr.csv"):
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def _validate_asr_output_dir(output_dir: Path) -> ValidationResult:
    if not output_dir.exists() or not output_dir.is_dir():
        return ValidationResult(False, [f"Output directory does not exist: {output_dir}"], [])

    transcript_files = sorted(output_dir.glob("*_transcripts.csv"))
    errors: list[str] = []
    if len(transcript_files) != 1:
        errors.append(
            "Expected exactly one *_transcripts.csv file, found "
            f"{len(transcript_files)} in {output_dir}"
        )
        return ValidationResult(False, errors, [])
    return _validate_transcript_file(transcript_files[0])


def run_single_video_pipeline(config: SingleVideoPipelineConfig) -> SingleVideoPipelineResult:
    output_dir = config.output_dir.expanduser().resolve()
    if config.dry_run:
        print(f"[dry-run] Target output directory: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    if config.video_id is not None:
        resolved_video_id = config.video_id
    elif config.dry_run:
        resolved_video_id = "dryrun_video_id"
    else:
        try:
            resolved_video_id = fetch_video_id(config.youtube_url, dry_run=False)
        except Exception as exc:
            raise PipelineStepError("resolve_video_id", str(exc)) from exc

    transcript_path = output_dir / f"{resolved_video_id}_transcripts.csv"
    segments_path = None if config.asr_only else output_dir / "segments.txt"
    existing_video_path = find_local_video_file(output_dir, resolved_video_id)

    if config.skip_existing and not config.dry_run:
        existing_validation = (
            _validate_asr_output_dir(output_dir)
            if config.asr_only
            else validate_meeting_dir(output_dir, require_ocr=True, min_slides=1)
        )
        if existing_validation.ok:
            result = _build_existing_bundle_result(
                output_dir=output_dir,
                resolved_video_id=resolved_video_id,
                existing_video_path=existing_video_path,
                validation_result=None if config.skip_validate else existing_validation,
                asr_only=config.asr_only,
            )
            if config.delete_video_after_processing and existing_video_path is not None:
                _delete_file_if_present(existing_video_path)
                result.video_path = None
            return result

    required_commands = ["yt-dlp", "ffmpeg"] if config.asr_only else ["yt-dlp", "ffmpeg", "tesseract"]
    require_commands(required_commands, dry_run=config.dry_run)

    reused_steps: list[str] = []

    try:
        if config.skip_existing and existing_video_path is not None:
            video_path = existing_video_path
            reused_steps.append("video")
        else:
            _, video_path = download_video(
                youtube_url=config.youtube_url,
                meeting_dir=output_dir,
                video_id=resolved_video_id,
                dry_run=config.dry_run,
            )
    except Exception as exc:
        raise PipelineStepError("download", str(exc)) from exc
    audio_source_path = output_dir / f"{resolved_video_id}.wav"

    result = SingleVideoPipelineResult(
        output_dir=output_dir,
        video_id=resolved_video_id,
        video_path=video_path,
        transcript_path=transcript_path,
        segments_path=segments_path,
        reused_steps=reused_steps,
    )

    transcript_valid = (not config.dry_run) and config.skip_existing and _transcript_is_valid(
        transcript_path
    )
    if transcript_valid:
        result.num_words = _count_csv_rows(transcript_path)
        result.reused_steps.append("transcript")
    else:
        try:
            extract_audio_to_wav(video_path, audio_source_path, dry_run=config.dry_run)
            result.num_words = transcribe_with_whisper(
                audio_path=audio_source_path,
                transcript_csv_path=transcript_path,
                model_name=config.whisper_model,
                language=config.language,
                dry_run=config.dry_run,
            )
        except Exception as exc:
            raise PipelineStepError("transcribe", str(exc)) from exc
        finally:
            if not config.keep_temp_audio and not config.dry_run and audio_source_path.exists():
                audio_source_path.unlink()

    if config.asr_only:
        if not config.skip_validate and not config.dry_run:
            result.validation_result = _validate_asr_output_dir(output_dir)
        should_delete_video = (
            config.delete_video_after_processing
            and not config.dry_run
            and (
                config.skip_validate
                or (result.validation_result is not None and result.validation_result.ok)
            )
        )
        if should_delete_video:
            _delete_file_if_present(result.video_path)
            result.video_path = None
        return result

    try:
        if not config.dry_run:
            _clear_segment_artifacts(output_dir)
        scene_end_times = detect_scene_end_times(
            video_path=video_path,
            threshold=config.scene_threshold,
            min_scene_len=config.min_scene_len,
            dry_run=config.dry_run,
        )
        if config.text_change_filter:
            filtered_end_times = filter_end_times_by_ocr_text_change(
                video_path=video_path,
                end_times=scene_end_times,
                similarity_threshold=config.text_sim_threshold,
                ocr_min_conf=config.ocr_min_conf,
                ocr_min_tokens=config.ocr_min_tokens,
                max_same_text_span=config.max_same_text_span,
                mask_top_ratio=config.ocr_mask_top_ratio,
                mask_bottom_ratio=config.ocr_mask_bottom_ratio,
                mask_left_ratio=config.ocr_mask_left_ratio,
                mask_right_ratio=config.ocr_mask_right_ratio,
                dry_run=config.dry_run,
            )
            if not config.dry_run:
                print(
                    "Text-change filter: "
                    f"{len(scene_end_times)} -> {len(filtered_end_times)} boundaries"
                )
            scene_end_times = filtered_end_times

        extracted_slide_paths = extract_slide_images(
            video_path=video_path,
            end_times=scene_end_times,
            meeting_dir=output_dir,
            dry_run=config.dry_run,
        )
        result.slide_paths, scene_end_times = dedupe_slide_images(
            extracted_slide_paths,
            scene_end_times,
            hamming_distance_threshold=config.slide_dedupe_hamming_distance,
            mask_top_ratio=config.ocr_mask_top_ratio,
            mask_bottom_ratio=config.ocr_mask_bottom_ratio,
            mask_left_ratio=config.ocr_mask_left_ratio,
            mask_right_ratio=config.ocr_mask_right_ratio,
            dry_run=config.dry_run,
        )
        result.num_segments = write_segments_txt(
            scene_end_times,
            segments_path=segments_path,
            dry_run=config.dry_run,
        )
        result.ocr_paths = run_ocr_on_slides(
            result.slide_paths,
            mask_top_ratio=config.ocr_mask_top_ratio,
            mask_bottom_ratio=config.ocr_mask_bottom_ratio,
            mask_left_ratio=config.ocr_mask_left_ratio,
            mask_right_ratio=config.ocr_mask_right_ratio,
            dry_run=config.dry_run,
        )
    except Exception as exc:
        raise PipelineStepError("segment", str(exc)) from exc

    if not config.skip_validate and not config.dry_run:
        result.validation_result = validate_meeting_dir(
            meeting_dir=output_dir,
            require_ocr=True,
            min_slides=1,
        )

    should_delete_video = (
        config.delete_video_after_processing
        and not config.dry_run
        and (
            config.skip_validate
            or (result.validation_result is not None and result.validation_result.ok)
        )
    )
    if should_delete_video:
        _delete_file_if_present(result.video_path)
        result.video_path = None

    return result


def build_parser() -> argparse.ArgumentParser:
    default_data_root = Path(__file__).resolve().parents[1] / "custom_data"
    parser = argparse.ArgumentParser(
        description=(
            "Download one YouTube lecture and generate TreeSeg-compatible "
            "artifacts under a local data root."
        )
    )
    parser.add_argument("--youtube-url", required=True, help="Single YouTube video URL.")
    parser.add_argument("--speaker", required=True, help="Speaker folder name.")
    parser.add_argument("--course-dir", required=True, help="Course directory name.")
    parser.add_argument("--meeting-id", required=True, help="Meeting/lecture ID folder.")
    parser.add_argument(
        "--data-root",
        default=str(default_data_root),
        help="Root output directory.",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        help="Local Whisper model name (default: base).",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language hint for Whisper (default: en).",
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=DEFAULT_SCENE_THRESHOLD,
        help=f"PySceneDetect content threshold (default: {DEFAULT_SCENE_THRESHOLD}).",
    )
    parser.add_argument(
        "--min-scene-len",
        type=float,
        default=DEFAULT_MIN_SCENE_LEN,
        help=f"Minimum scene length in seconds (default: {DEFAULT_MIN_SCENE_LEN}).",
    )
    parser.add_argument(
        "--text-change-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Filter scene boundaries using OCR text similarity so boundaries are "
            "kept mainly when slide text changes (enabled by default)."
        ),
    )
    parser.add_argument(
        "--text-sim-threshold",
        type=float,
        default=DEFAULT_TEXT_SIM_THRESHOLD,
        help=(
            "Jaccard similarity threshold for OCR text when --text-change-filter "
            f"is enabled (default: {DEFAULT_TEXT_SIM_THRESHOLD}). Higher keeps fewer boundaries."
        ),
    )
    parser.add_argument(
        "--ocr-min-conf",
        type=float,
        default=60.0,
        help="Minimum OCR confidence for text-change filtering (default: 60).",
    )
    parser.add_argument(
        "--ocr-min-tokens",
        type=int,
        default=6,
        help=(
            "Minimum token count before using OCR similarity for pruning "
            "(default: 6)."
        ),
    )
    parser.add_argument(
        "--max-same-text-span",
        type=float,
        default=900.0,
        help=(
            "Force a boundary if text appears unchanged for this many seconds "
            "(default: 900)."
        ),
    )
    parser.add_argument(
        "--keep-temp-audio",
        action="store_true",
        help="Keep extracted WAV file after transcription.",
    )
    parser.add_argument(
        "--delete-video-after-processing",
        action="store_true",
        help="Delete the downloaded video after successful preprocessing.",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip meeting directory validation at the end.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without running downloads/transcription.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = ensure_meeting_dir(
        data_root=data_root,
        speaker=args.speaker,
        course_dir=args.course_dir,
        meeting_id=args.meeting_id,
        dry_run=args.dry_run,
    )

    try:
        result = run_single_video_pipeline(
            SingleVideoPipelineConfig(
                youtube_url=args.youtube_url,
                output_dir=output_dir,
                whisper_model=args.whisper_model,
                language=args.language,
                scene_threshold=args.scene_threshold,
                min_scene_len=args.min_scene_len,
                text_change_filter=args.text_change_filter,
                text_sim_threshold=args.text_sim_threshold,
                ocr_min_conf=args.ocr_min_conf,
                ocr_min_tokens=args.ocr_min_tokens,
                max_same_text_span=args.max_same_text_span,
                keep_temp_audio=args.keep_temp_audio,
                delete_video_after_processing=args.delete_video_after_processing,
                skip_validate=args.skip_validate,
                dry_run=args.dry_run,
            )
        )
    except PipelineStepError as exc:
        print(f"Pipeline failed during {exc.stage}: {exc}")
        return 1

    if result.validation_result is not None:
        print_validation_result(result.validation_result)
        if not result.validation_result.ok:
            return 1

    print("Pipeline complete.")
    print(f"Meeting dir: {result.output_dir}")
    print(f"Video ID: {result.video_id}")
    print(f"Transcript: {result.transcript_path}")
    print(f"Segments: {result.segments_path} ({result.num_segments} boundaries)")
    print(f"Slides: {len(result.slide_paths)}")
    print(f"OCR CSVs: {len(result.ocr_paths)}")
    if result.num_words is not None:
        print(f"Transcribed words: {result.num_words}")
    if result.reused_steps:
        reused = ", ".join(result.reused_steps)
        print(f"Reused steps: {reused}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
