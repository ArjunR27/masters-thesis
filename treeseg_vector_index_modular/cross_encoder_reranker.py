from sentence_transformers import CrossEncoder

from .constants import SLIDE_TOKEN
from .device_resolver import DeviceResolver
from .rerank_input_builder import RerankInputBuilder


class CrossEncoderReranker:
    def __init__(self, model_name, device=None, input_builder=None):
        if device is None:
            device = DeviceResolver.resolve_device()
        self.model = CrossEncoder(model_name, device=device)
        self.input_builder = input_builder or RerankInputBuilder.build_rerank_input

    def rerank(self, query, results, top_n=5, slide_token=SLIDE_TOKEN):
        if not results:
            return []
        pairs = []
        for hit in results:
            rerank_text = self.input_builder(
                hit.get("text", ""), slide_token=slide_token
            )
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
