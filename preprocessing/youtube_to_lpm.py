from __future__ import annotations

import argparse
from pathlib import Path

from lpm_preprocess_lib import (
    detect_scene_end_times,
    download_video,
    ensure_meeting_dir,
    extract_audio_to_wav,
    extract_slide_images,
    filter_end_times_by_ocr_text_change,
    print_validation_result,
    require_commands,
    run_ocr_on_slides,
    transcribe_with_whisper,
    validate_meeting_dir,
    write_segments_txt,
)


def build_parser() -> argparse.ArgumentParser:
    default_data_root = Path(__file__).resolve().parents[1] / "custom_data"
    parser = argparse.ArgumentParser(
        description=(
            "Download one YouTube lecture and generate TreeSeg-compatible "
            "LPM artifacts under masters-thesis/lpm_data."
        )
    )
    parser.add_argument("--youtube-url", required=True, help="Single YouTube video URL.")
    parser.add_argument("--speaker", required=True, help="Speaker folder name.")
    parser.add_argument("--course-dir", required=True, help="Course directory name.")
    parser.add_argument("--meeting-id", required=True, help="Meeting/lecture ID folder.")
    parser.add_argument(
        "--data-root",
        default=str(default_data_root),
        help="Root lpm_data directory.",
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
        default=2.0,
        help="PySceneDetect content threshold (default: 2.0).",
    )
    parser.add_argument(
        "--min-scene-len",
        type=float,
        default=1.0,
        help="Minimum scene length in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--text-change-filter",
        action="store_true",
        help=(
            "Filter scene boundaries using OCR text similarity so boundaries are "
            "kept mainly when slide text changes."
        ),
    )
    parser.add_argument(
        "--text-sim-threshold",
        type=float,
        default=0.80,
        help=(
            "Jaccard similarity threshold for OCR text when --text-change-filter "
            "is enabled (default: 0.80). Higher keeps fewer boundaries."
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
    meeting_dir = ensure_meeting_dir(
        data_root=data_root,
        speaker=args.speaker,
        course_dir=args.course_dir,
        meeting_id=args.meeting_id,
        dry_run=args.dry_run,
    )

    require_commands(["yt-dlp", "ffmpeg", "tesseract"], dry_run=args.dry_run)

    video_id, video_path = download_video(
        youtube_url=args.youtube_url,
        meeting_dir=meeting_dir,
        dry_run=args.dry_run,
    )
    audio_path = meeting_dir / f"{video_id}.wav"
    transcript_path = meeting_dir / f"{video_id}_transcripts.csv"
    segments_path = meeting_dir / "segments.txt"

    extract_audio_to_wav(video_path, audio_path, dry_run=args.dry_run)
    num_words = transcribe_with_whisper(
        audio_path=audio_path,
        transcript_csv_path=transcript_path,
        model_name=args.whisper_model,
        language=args.language,
        dry_run=args.dry_run,
    )

    scene_end_times = detect_scene_end_times(
        video_path=video_path,
        threshold=args.scene_threshold,
        min_scene_len=args.min_scene_len,
        dry_run=args.dry_run,
    )
    if args.text_change_filter:
        filtered_end_times = filter_end_times_by_ocr_text_change(
            video_path=video_path,
            end_times=scene_end_times,
            similarity_threshold=args.text_sim_threshold,
            ocr_min_conf=args.ocr_min_conf,
            ocr_min_tokens=args.ocr_min_tokens,
            max_same_text_span=args.max_same_text_span,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            print(
                "Text-change filter: "
                f"{len(scene_end_times)} -> {len(filtered_end_times)} boundaries"
            )
        scene_end_times = filtered_end_times

    num_segments = write_segments_txt(
        scene_end_times,
        segments_path=segments_path,
        dry_run=args.dry_run,
    )
    slide_paths = extract_slide_images(
        video_path=video_path,
        end_times=scene_end_times,
        meeting_dir=meeting_dir,
        dry_run=args.dry_run,
    )
    ocr_paths = run_ocr_on_slides(slide_paths, dry_run=args.dry_run)

    if not args.keep_temp_audio and not args.dry_run and audio_path.exists():
        audio_path.unlink()

    if not args.skip_validate and not args.dry_run:
        result = validate_meeting_dir(meeting_dir=meeting_dir, require_ocr=True, min_slides=1)
        print_validation_result(result)
        if not result.ok:
            return 1

    print("Pipeline complete.")
    print(f"Meeting dir: {meeting_dir}")
    print(f"Video ID: {video_id}")
    print(f"Transcript: {transcript_path}")
    print(f"Segments: {segments_path} ({num_segments} boundaries)")
    print(f"Slides: {len(slide_paths)}")
    print(f"OCR CSVs: {len(ocr_paths)}")
    if not args.dry_run:
        print(f"Transcribed words: {num_words}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
