from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_CHUNK_STRATEGIES = ("utterance_packed", "raw_token_window")
DEFAULT_CHUNK_SIZES = (128, 256, 512)
DEFAULT_OVERLAP_PERCENTS = (0, 10)
DEFAULT_OCR_MODES = ("transcript_only", "combined_ocr")


@dataclass(frozen=True)
class BaselineRagConfig:
    chunk_strategy: str = "utterance_packed"
    chunk_size_tokens: int = 256
    overlap_percent: int = 0
    ocr_mode: str = "transcript_only"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize_embeddings: bool = True
    max_gap_s: str | float = "auto"
    lowercase: bool = True
    ocr_min_conf: float = 60.0
    ocr_per_slide: int = 1

    def __post_init__(self) -> None:
        if self.chunk_strategy not in DEFAULT_CHUNK_STRATEGIES:
            raise ValueError(f"Unsupported chunk_strategy: {self.chunk_strategy}")
        if int(self.chunk_size_tokens) <= 0:
            raise ValueError("chunk_size_tokens must be > 0")
        if int(self.overlap_percent) < 0 or int(self.overlap_percent) >= 100:
            raise ValueError("overlap_percent must be in [0, 100)")
        if self.ocr_mode not in DEFAULT_OCR_MODES:
            raise ValueError(f"Unsupported ocr_mode: {self.ocr_mode}")

    @property
    def attach_ocr(self) -> bool:
        return self.ocr_mode == "combined_ocr"

    @property
    def overlap_tokens(self) -> int:
        return int(round(self.chunk_size_tokens * (self.overlap_percent / 100.0)))

    @property
    def system_name(self) -> str:
        return (
            f"baseline__{self.chunk_strategy}"
            f"__{self.chunk_size_tokens}tok"
            f"__ov{self.overlap_percent}"
            f"__{self.ocr_mode}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_strategy": self.chunk_strategy,
            "chunk_size_tokens": self.chunk_size_tokens,
            "overlap_percent": self.overlap_percent,
            "overlap_tokens": self.overlap_tokens,
            "ocr_mode": self.ocr_mode,
            "embedding_model": self.embedding_model,
            "normalize_embeddings": self.normalize_embeddings,
            "max_gap_s": self.max_gap_s,
            "lowercase": self.lowercase,
            "ocr_min_conf": self.ocr_min_conf,
            "ocr_per_slide": self.ocr_per_slide,
        }


@dataclass(frozen=True)
class BaselineChunkRecord:
    chunk_id: int
    text: str
    start: float | None
    end: float | None
    lecture_key: str
    speaker: str
    course_dir: str
    meeting_id: str
    video_id: str
    chunk_strategy: str
    chunk_size_tokens: int
    overlap_tokens: int
    ocr_mode: str
    token_count: int
    source_item_start: int | None = None
    source_item_end: int | None = None
    slide_indices: tuple[int, ...] = field(default_factory=tuple)

    def to_payload(self) -> dict[str, object]:
        return {
            "index_kind": "baseline_chunk",
            "chunk_id": self.chunk_id,
            "segment_id": self.chunk_id,
            "node_id": None,
            "tree_path": None,
            "is_leaf": True,
            "depth": None,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "token_count": self.token_count,
            "chunk_strategy": self.chunk_strategy,
            "chunk_size_tokens": self.chunk_size_tokens,
            "overlap_tokens": self.overlap_tokens,
            "ocr_mode": self.ocr_mode,
            "source_item_start": self.source_item_start,
            "source_item_end": self.source_item_end,
            "ocr_slide_indices": list(self.slide_indices),
            "lecture_key": self.lecture_key,
            "speaker": self.speaker,
            "course_dir": self.course_dir,
            "meeting_id": self.meeting_id,
            "video_id": self.video_id,
        }


@dataclass(frozen=True)
class BaselineChunkBuildResult:
    records: list[dict[str, object]]
    skip_reason: str | None = None


@dataclass(frozen=True)
class BaselineStoreBuildResult:
    store: object
    skipped_lectures: dict[str, str]


def build_default_baseline_configs(
    chunk_strategies: list[str] | tuple[str, ...] | None = None,
    chunk_sizes: list[int] | tuple[int, ...] | None = None,
    overlap_percents: list[int] | tuple[int, ...] | None = None,
    ocr_modes: list[str] | tuple[str, ...] | None = None,
) -> list[BaselineRagConfig]:
    strategies = tuple(chunk_strategies or DEFAULT_CHUNK_STRATEGIES)
    sizes = tuple(chunk_sizes or DEFAULT_CHUNK_SIZES)
    overlaps = tuple(overlap_percents or DEFAULT_OVERLAP_PERCENTS)
    modes = tuple(ocr_modes or DEFAULT_OCR_MODES)

    configs = []
    for strategy in strategies:
        for size in sizes:
            for overlap in overlaps:
                for mode in modes:
                    configs.append(
                        BaselineRagConfig(
                            chunk_strategy=strategy,
                            chunk_size_tokens=int(size),
                            overlap_percent=int(overlap),
                            ocr_mode=mode,
                        )
                    )
    return configs
