import threading

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .device_resolver import DeviceResolver


class LpmVectorIndex:
    _shared_models = {}
    _shared_models_lock = threading.Lock()

    def __init__(self, model_name, device=None, normalize=True, build_global=True):
        if device is None:
            device = DeviceResolver.resolve_device()
        self.model = self._get_or_create_shared_model(model_name, device)
        self.normalize = normalize
        self.build_global = build_global
        self.lecture_indices = {}
        self.global_segments = []
        self.global_index = None
        self._global_embeddings = []
        self._global_segment_keys = []

    @classmethod
    def _shared_model_key(cls, model_name, device):
        return (str(model_name), str(device))

    @classmethod
    def _get_or_create_shared_model(cls, model_name, device):
        key = cls._shared_model_key(model_name, device)
        with cls._shared_models_lock:
            model = cls._shared_models.get(key)
            if model is None:
                model = SentenceTransformer(model_name, device=device)
                cls._shared_models[key] = model
            return model

    @classmethod
    def clear_shared_model_cache(cls):
        with cls._shared_models_lock:
            cls._shared_models = {}

    def _build_faiss_index(self, embeddings):
        dim = embeddings.shape[1]
        if self.normalize:
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index

    def _segment_lookup_key(self, segment):
        return (
            segment.get("lecture_key"),
            segment.get("tree_path"),
            segment.get("segment_id"),
            segment.get("slide_index"),
        )

    def _normalize_query_embedding(self, query_embedding):
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        return query_embedding

    def encode_query(self, query):
        embedding = self.model.encode([query], normalize_embeddings=self.normalize)
        return self._normalize_query_embedding(embedding)

    def _resolve_embeddings(self, segments):
        precomputed = [segment.get("embedding") for segment in segments]
        if all(embedding is not None for embedding in precomputed):
            embeddings = np.asarray(precomputed, dtype=np.float32)
        else:
            texts = [segment["text"] for segment in segments]
            embeddings = self.model.encode(texts, normalize_embeddings=self.normalize)
            embeddings = np.asarray(embeddings, dtype=np.float32)
        return embeddings

    def add_lecture(self, lecture, segments):
        if not segments:
            return
        embeddings = self._resolve_embeddings(segments)
        index = self._build_faiss_index(embeddings)

        segment_key_to_idx = {}
        segments_by_tree_path = {}
        for idx, segment in enumerate(segments):
            segment_key_to_idx[self._segment_lookup_key(segment)] = idx
            tree_path = segment.get("tree_path")
            if tree_path:
                segments_by_tree_path[tree_path] = segment

        self.lecture_indices[lecture.key] = {
            "lecture": lecture,
            "segments": segments,
            "embeddings": embeddings,
            "index": index,
            "segment_key_to_idx": segment_key_to_idx,
            "segments_by_tree_path": segments_by_tree_path,
        }

        if self.build_global:
            self._global_embeddings.append(embeddings)
            self.global_segments.extend(segments)
            self._global_segment_keys.extend(
                [self._segment_lookup_key(segment) for segment in segments]
            )

    def finalize(self):
        if not self.build_global or not self._global_embeddings:
            return
        all_embeddings = np.vstack(self._global_embeddings).astype(np.float32)
        self.global_index = self._build_faiss_index(all_embeddings)
        self._global_embeddings = []

    def _clone_result_payload(self, segment):
        payload = {
            key: value for key, value in segment.items() if key != "embedding"
        }
        return payload

    def _search_entry(self, index, segments, query_embedding, top_k):
        if index is None:
            raise ValueError("Index not built yet.")
        if not segments:
            return []
        k = min(top_k, len(segments))
        scores, idxs = index.search(query_embedding, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            segment = segments[idx]
            payload = self._clone_result_payload(segment)
            payload["score"] = float(score)
            results.append(payload)
        return results

    def search_with_embedding(self, query_embedding, top_k=5, lecture_key=None):
        query_embedding = self._normalize_query_embedding(query_embedding)
        if lecture_key:
            entry = self.lecture_indices.get(lecture_key)
            if entry is None:
                raise KeyError(f"Unknown lecture key: {lecture_key}")
            return self._search_entry(
                entry["index"], entry["segments"], query_embedding, top_k=top_k
            )
        return self._search_entry(
            self.global_index, self.global_segments, query_embedding, top_k=top_k
        )

    def search(self, query, top_k=5, lecture_key=None):
        query_embedding = self.encode_query(query)
        return self.search_with_embedding(
            query_embedding, top_k=top_k, lecture_key=lecture_key
        )

    def _get_lecture_entry_for_hit(self, hit, lecture_key=None):
        resolved_lecture_key = lecture_key or hit.get("lecture_key")
        if not resolved_lecture_key:
            raise ValueError("lecture_key is required to expand summary-tree hits.")
        entry = self.lecture_indices.get(resolved_lecture_key)
        if entry is None:
            raise KeyError(f"Unknown lecture key: {resolved_lecture_key}")
        return entry

    def _score_embeddings(self, query_embedding, embeddings):
        query_embedding = self._normalize_query_embedding(query_embedding)
        if self.normalize:
            scores = embeddings @ query_embedding[0]
        else:
            deltas = embeddings - query_embedding[0]
            scores = -np.sum(np.square(deltas), axis=1)
        return np.asarray(scores, dtype=np.float32)

    def _lookup_supporting_leaf_hits(
        self, entry, descendant_leaf_tree_paths, query_embedding, top_descendant_leaves
    ):
        if not descendant_leaf_tree_paths:
            return []
        positions = []
        for tree_path in descendant_leaf_tree_paths:
            segment = entry["segments_by_tree_path"].get(tree_path)
            if segment is None:
                continue
            key = self._segment_lookup_key(segment)
            idx = entry["segment_key_to_idx"].get(key)
            if idx is None:
                continue
            positions.append((idx, segment))
        if not positions:
            return []

        embeddings = np.asarray(
            [entry["embeddings"][idx] for idx, _ in positions], dtype=np.float32
        )
        scores = self._score_embeddings(query_embedding, embeddings)
        order = np.argsort(-scores)
        supporting_hits = []
        for rank_idx in order[: min(top_descendant_leaves, len(order))]:
            segment_idx, segment = positions[int(rank_idx)]
            hit = self._clone_result_payload(segment)
            hit["score"] = float(scores[int(rank_idx)])
            hit["support_rank"] = len(supporting_hits) + 1
            hit["support_source"] = "descendant_leaf"
            hit["index_position"] = segment_idx
            supporting_hits.append(hit)
        return supporting_hits

    def expand_summary_tree_results(
        self,
        query,
        results,
        lecture_key=None,
        top_descendant_leaves=3,
        query_embedding=None,
    ):
        if not results:
            return []
        if query_embedding is None:
            query_embedding = self.encode_query(query)
        else:
            query_embedding = self._normalize_query_embedding(query_embedding)

        expanded = []
        for hit in results:
            if hit.get("index_kind") != "summary_tree":
                expanded.append(dict(hit))
                continue

            updated = dict(hit)
            descendant_leaf_tree_paths = list(
                updated.get("descendant_leaf_tree_paths") or []
            )
            if updated.get("is_leaf", True):
                updated["supporting_leaves"] = []
                if not descendant_leaf_tree_paths and updated.get("tree_path"):
                    updated["descendant_leaf_tree_paths"] = [updated["tree_path"]]
                expanded.append(updated)
                continue

            entry = self._get_lecture_entry_for_hit(updated, lecture_key=lecture_key)
            updated["supporting_leaves"] = self._lookup_supporting_leaf_hits(
                entry,
                descendant_leaf_tree_paths,
                query_embedding,
                top_descendant_leaves=top_descendant_leaves,
            )
            expanded.append(updated)
        return expanded

    @staticmethod
    def _tree_overlap(hit_a, hit_b):
        path_a = hit_a.get("tree_path") or ""
        path_b = hit_b.get("tree_path") or ""
        if not path_a or not path_b:
            return False
        return path_a.startswith(path_b) or path_b.startswith(path_a)

    @staticmethod
    def _descendant_overlap(hit_a, hit_b):
        leaves_a = set(hit_a.get("descendant_leaf_tree_paths") or [])
        leaves_b = set(hit_b.get("descendant_leaf_tree_paths") or [])
        if not leaves_a or not leaves_b:
            return False
        return bool(leaves_a.intersection(leaves_b))

    def deduplicate_summary_tree_results(self, results):
        deduped = []
        for hit in results:
            if hit.get("index_kind") != "summary_tree":
                deduped.append(hit)
                continue
            if any(
                kept.get("index_kind") == "summary_tree"
                and kept.get("lecture_key") == hit.get("lecture_key")
                and (
                    self._tree_overlap(kept, hit)
                    or self._descendant_overlap(kept, hit)
                )
                for kept in deduped
            ):
                continue
            deduped.append(hit)
        return deduped
