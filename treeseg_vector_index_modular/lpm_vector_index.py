import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .device_resolver import DeviceResolver


class LpmVectorIndex:
    def __init__(self, model_name, device=None, normalize=True, build_global=True):
        if device is None:
            device = DeviceResolver.resolve_device()
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
                    "slide_index": seg.get("slide_index"),
                    "modality": seg.get("modality"),
                    "tree_path": seg.get("tree_path"),
                    "utterance_start": seg.get("utterance_start"),
                    "utterance_end": seg.get("utterance_end"),
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": seg.get("text"),
                }
            )
        return results
