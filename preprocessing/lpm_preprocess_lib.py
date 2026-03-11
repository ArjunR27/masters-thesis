from __future__ import annotations

import csv
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence
import torch
from collections import Counter

import cv2
import pandas as pd
import pytesseract
import whisper
from pytesseract import Output as TesseractOutput
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

device = "cpu"


TRANSCRIPT_COLUMNS = ["Word", "Start", "End"]
OCR_REQUIRED_COLUMNS = [
    "level",
    "page_num",
    "block_num",
    "par_num",
    "line_num",
    "word_num",
    "left",
    "top",
    "width",
    "height",
    "conf",
    "text",
]

SLIDE_IMAGE_RE = re.compile(r"^slide_(\d+)\.jpg$", re.IGNORECASE)
SLIDE_OCR_RE = re.compile(r"^slide_(\d+)_ocr\.csv$", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _run(
    cmd: Sequence[str],
    *,
    dry_run: bool = False,
    capture_output: bool = False,   
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str] | None:
    cmd_text = " ".join(cmd)
    print(f"$ {cmd_text}")
    if dry_run:
        return None
    return subprocess.run(
        list(cmd),
        check=True,
        text=True,
        capture_output=capture_output,
        cwd=str(cwd) if cwd else None,
    )


def require_commands(commands: Iterable[str], *, dry_run: bool = False) -> None:
    missing = [name for name in commands if shutil.which(name) is None]
    if missing and not dry_run:
        joined = ", ".join(sorted(missing))
        raise RuntimeError(
            f"Missing required system commands: {joined}. "
            "Install them before running preprocessing."
        )
    if missing and dry_run:
        joined = ", ".join(sorted(missing))
        print(f"[dry-run] Missing commands detected (not blocking): {joined}")


def ensure_meeting_dir(
    data_root: Path,
    speaker: str,
    course_dir: str,
    meeting_id: str,
    *,
    dry_run: bool = False,
) -> Path:
    meeting_dir = data_root / speaker / course_dir / meeting_id
    if dry_run:
        print(f"[dry-run] Would create directory: {meeting_dir}")
    else:
        meeting_dir.mkdir(parents=True, exist_ok=True)
    return meeting_dir


def fetch_video_id(youtube_url: str, *, dry_run: bool = False) -> str:
    if dry_run:
        return "dryrun_video_id"
    proc = _run(
        ["yt-dlp", "--no-playlist", "--get-id", youtube_url],
        dry_run=False,
        capture_output=True,
    )
    assert proc is not None
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Failed to resolve YouTube video ID from URL.")
    return lines[0]


def download_video(
    youtube_url: str,
    meeting_dir: Path,
    *,
    video_id: str | None = None,
    dry_run: bool = False,
) -> tuple[str, Path]:
    resolved_video_id = video_id or fetch_video_id(youtube_url, dry_run=dry_run)
    output_template = str(meeting_dir / f"{resolved_video_id}.%(ext)s")
    _run(
        [
            "yt-dlp",
            "--no-playlist",
            "-f",
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format",
            "mp4",
            "-o",
            output_template,
            youtube_url,
        ],
        dry_run=dry_run,
    )
    expected_path = meeting_dir / f"{resolved_video_id}.mp4"
    if dry_run:
        return resolved_video_id, expected_path

    if expected_path.exists():
        return resolved_video_id, expected_path

    candidates = sorted(
        p
        for p in meeting_dir.glob(f"{resolved_video_id}.*")
        if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}
    )
    if not candidates:
        raise RuntimeError(
            "yt-dlp completed but no local video file was found for "
            f"{resolved_video_id}."
        )
    return resolved_video_id, candidates[0]


def extract_audio_to_wav(
    video_path: Path,
    audio_path: Path,
    *,
    dry_run: bool = False,
) -> Path:
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(audio_path),
        ],
        dry_run=dry_run,
    )
    return audio_path


def transcribe_with_whisper(
    audio_path: Path,
    transcript_csv_path: Path,
    *,
    model_name: str = "base",
    language: str = "en",
    dry_run: bool = False,
) -> int:
    if dry_run:
        print(
            "[dry-run] Would run Whisper transcription and write "
            f"{transcript_csv_path}"
        )
        return 0

    model = whisper.load_model(model_name, device=device)
    result = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        task="transcribe",
        verbose=False,
    )

    rows: list[dict[str, float | str]] = []
    for segment in result.get("segments", []) or []:
        for word in segment.get("words", []) or []:
            token = str(word.get("word", "")).strip()
            start = word.get("start")
            end = word.get("end")
            if not token or start is None or end is None:
                continue
            rows.append(
                {
                    "Word": re.sub(r"\s+", " ", token),
                    "Start": round(float(start), 4),
                    "End": round(float(end), 4),
                }
            )

    if not rows:
        raise RuntimeError(
            "Whisper completed but returned no word-level timestamps. "
            "Ensure `word_timestamps=True` is supported by the selected model."
        )

    df = pd.DataFrame(rows, columns=TRANSCRIPT_COLUMNS)
    transcript_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(transcript_csv_path, index=False)
    return len(df)


def get_video_duration_seconds(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps and frame_count and fps > 0:
        return float(frame_count / fps)
    return 0.0


def detect_scene_end_times(
    video_path: Path,
    *,
    threshold: float = 2.0,
    min_scene_len: float = 1.0,
    dry_run: bool = False,
) -> list[float]:
    if dry_run:
        print(
            "[dry-run] Would run scene detection and write segments.txt "
            f"for {video_path}"
        )
        return [30.0]

    video = open_video(str(video_path))
    fps = float(getattr(video, "frame_rate", 30.0) or 30.0)
    min_scene_len_frames = max(1, int(round(min_scene_len * fps)))
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold, min_scene_len=min_scene_len_frames)
    )
    scene_manager.detect_scenes(video=video)
    scene_list = scene_manager.get_scene_list()

    if scene_list:
        end_times = [float(end.get_seconds()) for _, end in scene_list]
    else:
        duration = get_video_duration_seconds(video_path)
        if duration <= 0:
            raise RuntimeError(
                "Scene detection found no scenes and video duration could not be read."
            )
        end_times = [duration]

    return normalize_end_times(end_times)


def _ocr_tokens_from_frame(
    frame,
    *,
    min_conf: float = 60.0,
    min_token_len: int = 2,
) -> set[str]:
    if frame is None:
        return set()

    # Light preprocessing improves OCR robustness on low-contrast slides.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    df = pytesseract.image_to_data(bw, output_type=TesseractOutput.DATAFRAME)
    if df is None or df.empty:
        return set()

    tokens = []
    for _, row in df.iterrows():
        text = str(row.get("text", "") or "").strip()
        if not text:
            continue

        conf_val = row.get("conf")
        try:
            conf = float(conf_val)
        except (TypeError, ValueError):
            conf = -1.0
        if conf < min_conf:
            continue

        for tok in _TOKEN_RE.findall(text.lower()):
            if len(tok) >= min_token_len:
                tokens.append(tok)

    if not tokens:
        return set()

    # Keep tokens with frequency >=1 as a set for Jaccard similarity.
    token_counts = Counter(tokens)
    return {tok for tok, cnt in token_counts.items() if cnt >= 1}


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / float(len(union))


def filter_end_times_by_ocr_text_change(
    video_path: Path,
    end_times: Sequence[float],
    *,
    similarity_threshold: float = 0.80,
    ocr_min_conf: float = 60.0,
    ocr_min_token_len: int = 2,
    ocr_min_tokens: int = 6,
    max_same_text_span: float = 900.0,
    dry_run: bool = False,
) -> list[float]:
    """
    Reduce over-segmentation by keeping boundaries only when OCR text changes.

    We still preserve the final end time so the last segment reaches the end.
    """
    cleaned = normalize_end_times(end_times)
    if len(cleaned) <= 1:
        return cleaned

    if dry_run:
        print(
            "[dry-run] Would apply OCR text-change boundary filter "
            f"(threshold={similarity_threshold}, min_conf={ocr_min_conf})."
        )
        return cleaned

    midpoints = compute_segment_midpoints(cleaned)
    if len(midpoints) != len(cleaned):
        return cleaned

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("[warn] Failed to open video for OCR text filtering; keeping scene boundaries.")
        return cleaned

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not fps or fps <= 0:
        fps = 30.0
    max_frame_idx = max(0, int(frame_count) - 1) if frame_count else None

    kept_end_times: list[float] = []
    prev_tokens: set[str] | None = None
    last_kept_end = 0.0

    for idx, (midpoint, end_time) in enumerate(zip(midpoints, cleaned)):
        frame_idx = max(0, int(round(midpoint * fps)))
        if max_frame_idx is not None:
            frame_idx = min(frame_idx, max_frame_idx)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        tokens = _ocr_tokens_from_frame(
            frame if ok else None,
            min_conf=ocr_min_conf,
            min_token_len=ocr_min_token_len,
        )

        if idx == 0:
            kept_end_times.append(end_time)
            prev_tokens = tokens
            last_kept_end = end_time
            continue

        # Always keep final boundary to preserve total duration.
        if idx == len(cleaned) - 1:
            kept_end_times.append(end_time)
            prev_tokens = tokens
            last_kept_end = end_time
            continue

        token_rich_prev = prev_tokens is not None and len(prev_tokens) >= ocr_min_tokens
        token_rich_curr = len(tokens) >= ocr_min_tokens
        similar = False

        if token_rich_prev and token_rich_curr:
            similarity = _jaccard_similarity(prev_tokens or set(), tokens)
            similar = similarity >= similarity_threshold
        else:
            # If OCR is weak in either frame, avoid aggressive pruning.
            similar = False

        span = end_time - last_kept_end
        force_keep_for_span = span >= max_same_text_span

        if (not similar) or force_keep_for_span:
            kept_end_times.append(end_time)
            prev_tokens = tokens
            last_kept_end = end_time

    cap.release()
    return normalize_end_times(kept_end_times)


def normalize_end_times(end_times: Iterable[float], *, eps: float = 1e-6) -> list[float]:
    cleaned: list[float] = []
    last = -1.0
    for value in end_times:
        numeric = float(value)
        if numeric <= 0:
            continue
        if numeric <= last + eps:
            continue
        cleaned.append(numeric)
        last = numeric
    return cleaned


def write_segments_txt(
    end_times: Iterable[float],
    segments_path: Path,
    *,
    dry_run: bool = False,
) -> int:
    cleaned = normalize_end_times(end_times)
    if not cleaned:
        raise RuntimeError("No valid slide end times available for segments.txt.")
    if dry_run:
        print(f"[dry-run] Would write {len(cleaned)} segment boundaries to {segments_path}")
        return len(cleaned)
    segments_path.parent.mkdir(parents=True, exist_ok=True)
    with segments_path.open("w", encoding="utf-8") as f:
        for value in cleaned:
            f.write(f"{value:.4f}\n")
    return len(cleaned)


def compute_segment_midpoints(end_times: Sequence[float]) -> list[float]:
    if not end_times:
        return []
    starts = [0.0] + [float(t) for t in end_times[:-1]]
    mids = []
    for start, end in zip(starts, end_times):
        mids.append((float(start) + float(end)) / 2.0)
    return mids


def extract_slide_images(
    video_path: Path,
    end_times: Sequence[float],
    meeting_dir: Path,
    *,
    dry_run: bool = False,
) -> list[Path]:
    midpoints = compute_segment_midpoints(end_times)
    if not midpoints:
        raise RuntimeError("Cannot extract slide images without segment boundaries.")

    if dry_run:
        max_seek = None
    else:
        duration = get_video_duration_seconds(video_path)
        max_seek = max(0.0, duration - 0.05) if duration > 0 else None

    image_paths: list[Path] = []
    for idx, midpoint in enumerate(midpoints):
        if max_seek is not None:
            seek = min(max(midpoint, 0.0), max_seek)
        else:
            seek = max(midpoint, 0.0)

        image_path = meeting_dir / f"slide_{idx:03d}.jpg"
        image_paths.append(image_path)
        _run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{seek:.4f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(image_path),
            ],
            dry_run=dry_run,
        )
    return image_paths


def run_ocr_on_slides(slide_image_paths: Sequence[Path], *, dry_run: bool = False) -> list[Path]:
    ocr_csv_paths: list[Path] = []
    for image_path in slide_image_paths:
        match = SLIDE_IMAGE_RE.match(image_path.name)
        if not match:
            continue
        slide_idx = int(match.group(1))
        out_path = image_path.parent / f"slide_{slide_idx:03d}_ocr.csv"
        ocr_csv_paths.append(out_path)

        if dry_run:
            print(f"[dry-run] Would OCR {image_path} -> {out_path}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read slide image for OCR: {image_path}")
        df = pytesseract.image_to_data(image, output_type=TesseractOutput.DATAFRAME)
        if df is None:
            df = pd.DataFrame(columns=OCR_REQUIRED_COLUMNS)

        for column in OCR_REQUIRED_COLUMNS:
            if column not in df.columns:
                df[column] = ""

        df = df[OCR_REQUIRED_COLUMNS]
        df["text"] = df["text"].fillna("").astype(str)
        df["conf"] = pd.to_numeric(df["conf"], errors="coerce")
        df.to_csv(out_path, index=False)

    return ocr_csv_paths


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _read_csv_header(csv_path: Path) -> list[str]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return []


def _parse_segments(segments_path: Path) -> tuple[list[float], list[str]]:
    values: list[float] = []
    errors: list[str] = []
    with segments_path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                value = float(text)
            except ValueError:
                errors.append(f"segments.txt line {lineno} is not numeric: {text!r}")
                continue
            values.append(value)
    for idx in range(1, len(values)):
        if values[idx] <= values[idx - 1]:
            errors.append(
                "segments.txt must be strictly increasing but "
                f"line {idx + 1} <= line {idx}"
            )
            break
    return values, errors


def _contiguous_from_zero(indices: Sequence[int]) -> bool:
    if not indices:
        return False
    return sorted(indices) == list(range(min(indices), max(indices) + 1)) and min(indices) == 0


def validate_meeting_dir(
    meeting_dir: Path,
    *,
    require_ocr: bool = True,
    min_slides: int = 1,
) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if not meeting_dir.exists() or not meeting_dir.is_dir():
        return ValidationResult(False, [f"Meeting directory does not exist: {meeting_dir}"], [])

    transcript_files = sorted(meeting_dir.glob("*_transcripts.csv"))
    if len(transcript_files) != 1:
        errors.append(
            "Expected exactly one *_transcripts.csv file, found "
            f"{len(transcript_files)} in {meeting_dir}"
        )
    else:
        header = _read_csv_header(transcript_files[0])
        if header != TRANSCRIPT_COLUMNS:
            errors.append(
                f"Transcript header must be exactly {TRANSCRIPT_COLUMNS}, found {header}"
            )

    segments_path = meeting_dir / "segments.txt"
    if not segments_path.exists():
        errors.append(f"Missing required file: {segments_path}")
        segment_values: list[float] = []
    else:
        segment_values, segment_errors = _parse_segments(segments_path)
        errors.extend(segment_errors)
        if not segment_values:
            errors.append("segments.txt has no valid segment end times.")

    slide_images = []
    slide_image_indices: list[int] = []
    for path in sorted(meeting_dir.glob("slide_*.jpg")):
        match = SLIDE_IMAGE_RE.match(path.name)
        if not match:
            continue
        slide_images.append(path)
        slide_image_indices.append(int(match.group(1)))

    if slide_images and not _contiguous_from_zero(slide_image_indices):
        errors.append("Slide image indices are not contiguous starting from 0.")
    if not slide_images:
        warnings.append("No slide_###.jpg files found. Continuing without image checks.")
    elif len(slide_images) < min_slides:
        warnings.append(
            "Found fewer slide images than expected: "
            f"{len(slide_images)} < {min_slides}"
        )

    if segment_values and slide_images and len(segment_values) != len(slide_images):
        errors.append(
            "segments.txt entry count must match number of slide images: "
            f"{len(segment_values)} vs {len(slide_images)}"
        )

    ocr_files = []
    ocr_indices: list[int] = []
    for path in sorted(meeting_dir.glob("slide_*_ocr.csv")):
        match = SLIDE_OCR_RE.match(path.name)
        if not match:
            continue
        ocr_files.append(path)
        ocr_indices.append(int(match.group(1)))

    if require_ocr:
        if not ocr_files or len(ocr_files) < min_slides:
            errors.append(
                "Expected at least "
                f"{min_slides} OCR files named slide_###_ocr.csv, found {len(ocr_files)}"
            )
        if ocr_files and not _contiguous_from_zero(ocr_indices):
            errors.append("OCR indices are not contiguous starting from 0.")
        if slide_images and ocr_files and len(slide_images) != len(ocr_files):
            errors.append(
                "Slide image count and OCR file count must match: "
                f"{len(slide_images)} vs {len(ocr_files)}"
            )
        for ocr_path in ocr_files:
            header = _read_csv_header(ocr_path)
            missing = [c for c in OCR_REQUIRED_COLUMNS if c not in header]
            if missing:
                errors.append(f"{ocr_path.name} is missing OCR columns: {missing}")

    if not errors and not transcript_files:
        warnings.append("No transcript file found.")
    return ValidationResult(ok=not errors, errors=errors, warnings=warnings)


def print_validation_result(result: ValidationResult) -> None:
    if result.ok:
        print("Validation passed.")
    else:
        print("Validation failed.")
    for warning in result.warnings:
        print(f"Warning: {warning}")
    for error in result.errors:
        print(f"Error: {error}")
