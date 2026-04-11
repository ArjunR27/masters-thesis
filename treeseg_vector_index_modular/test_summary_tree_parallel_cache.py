import hashlib
import importlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import treeseg_vector_index_modular.lecture_segment_builder as lecture_segment_builder_module
import treeseg_vector_index_modular.lpm_vector_index as lpm_vector_index_module

from treeseg_vector_index_modular.lecture_descriptor import LectureDescriptor
from treeseg_vector_index_modular.lecture_segment_builder import SummaryTreeBuildOptions
from treeseg_vector_index_modular.lpm_config_builder import LpmConfigBuilder
from treeseg_vector_index_modular.vector_store_factory import VectorStoreFactory


class FakeSentenceTransformer:
    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device

    @staticmethod
    def _encode_one(text):
        digest = hashlib.sha256((text or "").encode("utf-8")).digest()
        return np.frombuffer(digest[:32], dtype=np.uint8).astype(np.float32)

    def encode(
        self,
        texts,
        batch_size=None,
        convert_to_numpy=False,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        del batch_size, show_progress_bar
        single_input = isinstance(texts, str)
        values = [texts] if single_input else list(texts)
        rows = []
        for value in values:
            vector = self._encode_one(value)
            if normalize_embeddings:
                norm = np.linalg.norm(vector)
                if norm:
                    vector = vector / norm
            rows.append(vector.astype(np.float32))
        array = np.vstack(rows).astype(np.float32)
        if not convert_to_numpy:
            result = [row.tolist() for row in array]
            return result[0] if single_input else result
        return array[0] if single_input else array


class MpsFailingSentenceTransformer(FakeSentenceTransformer):
    def __init__(self, model_name, device=None):
        if device == "mps":
            raise NotImplementedError(
                "Cannot copy out of meta tensor; no data! Please use "
                "torch.nn.Module.to_empty() instead of torch.nn.Module.to() "
                "when moving module from meta to a different device."
            )
        super().__init__(model_name, device=device)


class MockSummaryResponder:
    call_count = 0

    @classmethod
    def reset(cls):
        cls.call_count = 0

    @staticmethod
    def generate_summary(
        text,
        is_leaf=True,
        model="llama3.2",
        temperature=0.2,
        keep_alive=None,
        client=None,
        host=None,
    ):
        del is_leaf, model, temperature, keep_alive, client, host
        MockSummaryResponder.call_count += 1
        cleaned = " ".join((text or "").split())
        return cleaned[:220] or "<blank>"


def build_temp_lectures(video_ids, temp_root):
    source_root = (
        Path(__file__).resolve().parents[1]
        / "eduvid_evaluation"
        / "storage"
        / "videos"
    )
    lectures = []
    for video_id in video_ids:
        source_path = source_root / video_id / f"{video_id}_transcripts.csv"
        target_dir = temp_root / video_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / source_path.name
        shutil.copy2(source_path, target_path)
        lectures.append(
            LectureDescriptor(
                speaker="eduvid",
                course_dir="real_world_test",
                meeting_id=video_id,
                video_id=video_id,
                transcripts_path=str(target_path),
                meeting_dir=str(target_dir),
            )
        )
    return lectures


def canonicalize_segments(segments):
    canonical = []
    for segment in segments:
        item = {}
        for key, value in segment.items():
            if key == "embedding" and value is not None:
                array = np.asarray(value, dtype=np.float32).reshape(-1)
                item[key] = [round(float(x), 6) for x in array.tolist()]
            elif isinstance(value, float):
                item[key] = round(value, 6)
            else:
                item[key] = value
        canonical.append(item)
    canonical.sort(
        key=lambda record: (
            str(record.get("lecture_key")),
            str(record.get("tree_path")),
            -1 if record.get("segment_id") is None else int(record["segment_id"]),
        )
    )
    return canonical


def extract_segments_by_lecture(store):
    return {
        lecture_key: canonicalize_segments(entry["segments"])
        for lecture_key, entry in store.lecture_indices.items()
    }


def build_store(
    lectures,
    cache_dir,
    workers,
    rebuild_cache,
    treeseg_device="cpu",
    treeseg_transformer_cls=FakeSentenceTransformer,
):
    treeseg_module = importlib.import_module("treeseg.treeseg")
    treeseg_module.SentenceTransformer = treeseg_transformer_cls
    lpm_vector_index_module.SentenceTransformer = FakeSentenceTransformer
    lecture_segment_builder_module.OllamaResponder = MockSummaryResponder
    treeseg_module.TreeSeg._worker_local = __import__("threading").local()

    config = LpmConfigBuilder.build_lpm_config(
        embedding_model="fake-summary-tree-model",
        device=treeseg_device,
        batch_size=16,
        normalize=True,
    )
    factory = VectorStoreFactory()
    return factory.build_vector_store(
        lectures=lectures,
        treeseg_config=config,
        embed_model="fake-summary-tree-model",
        device="cpu",
        normalize=True,
        build_global=False,
        max_gap_s="auto",
        lowercase=True,
        attach_ocr=False,
        include_ocr_in_treeseg=False,
        ocr_min_conf=60.0,
        ocr_per_slide=1,
        target_segments=None,
        index_kind="summary_tree",
        summary_tree_build_options=SummaryTreeBuildOptions(
            workers=workers,
            cache_dir=str(cache_dir),
            rebuild_cache=rebuild_cache,
            ollama_host=None,
            cache_version="v1",
        ),
    )


def main():
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig-"))

    with tempfile.TemporaryDirectory(prefix="summary-tree-cache-test-") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        lectures = build_temp_lectures(
            ["DmBw8SEeAc0", "NVH8EYPHi30", "H-HVZJ7kGI0"],
            temp_dir / "lectures",
        )

        sequential_cache = temp_dir / "cache-sequential"
        parallel_cache = temp_dir / "cache-parallel"

        MockSummaryResponder.reset()
        sequential_store = build_store(
            lectures,
            cache_dir=sequential_cache,
            workers=1,
            rebuild_cache=True,
        )
        sequential_calls = MockSummaryResponder.call_count
        if sequential_calls <= 0:
            raise SystemExit("Expected summary calls during cold sequential build.")

        MockSummaryResponder.reset()
        parallel_store = build_store(
            lectures,
            cache_dir=parallel_cache,
            workers=2,
            rebuild_cache=True,
        )
        parallel_calls = MockSummaryResponder.call_count
        if parallel_calls <= 0:
            raise SystemExit("Expected summary calls during cold parallel build.")

        expected_order = [lecture.key for lecture in lectures]
        if list(sequential_store.lecture_indices.keys()) != expected_order:
            raise SystemExit("Sequential lecture merge order was not deterministic.")
        if list(parallel_store.lecture_indices.keys()) != expected_order:
            raise SystemExit("Parallel lecture merge order was not deterministic.")

        sequential_segments = extract_segments_by_lecture(sequential_store)
        parallel_segments = extract_segments_by_lecture(parallel_store)
        if sequential_segments != parallel_segments:
            raise SystemExit("Sequential and parallel summary-tree records differ.")

        cache_files = sorted(parallel_cache.glob("*.pkl.gz"))
        if len(cache_files) != len(lectures):
            raise SystemExit("Expected one cache file per lecture after cold build.")

        MockSummaryResponder.reset()
        warm_store = build_store(
            lectures,
            cache_dir=parallel_cache,
            workers=2,
            rebuild_cache=False,
        )
        if MockSummaryResponder.call_count != 0:
            raise SystemExit("Warm cache run should not invoke summary generation.")
        warm_segments = extract_segments_by_lecture(warm_store)
        if warm_segments != parallel_segments:
            raise SystemExit("Warm cache records differ from cold parallel build.")

        first_transcript = Path(lectures[0].transcripts_path)
        current_stat = first_transcript.stat()
        os.utime(
            first_transcript,
            ns=(current_stat.st_atime_ns, current_stat.st_mtime_ns + 1_000_000),
        )

        MockSummaryResponder.reset()
        invalidated_store = build_store(
            lectures,
            cache_dir=parallel_cache,
            workers=2,
            rebuild_cache=False,
        )
        if MockSummaryResponder.call_count <= 0:
            raise SystemExit("Cache invalidation should trigger a rebuild.")
        invalidated_segments = extract_segments_by_lecture(invalidated_store)
        if invalidated_segments != parallel_segments:
            raise SystemExit("Cache invalidation changed summary-tree records.")

        MockSummaryResponder.reset()
        fallback_store = build_store(
            lectures,
            cache_dir=temp_dir / "cache-parallel-mps-fallback",
            workers=2,
            rebuild_cache=True,
            treeseg_device="mps",
            treeseg_transformer_cls=MpsFailingSentenceTransformer,
        )
        fallback_segments = extract_segments_by_lecture(fallback_store)
        if fallback_segments != parallel_segments:
            raise SystemExit(
                "Parallel summary-tree CPU fallback changed the built records."
            )

    print("Summary-tree parallel/cache smoke test passed.")


if __name__ == "__main__":
    main()
