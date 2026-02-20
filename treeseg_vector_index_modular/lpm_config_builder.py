from .device_resolver import DeviceResolver


class LpmConfigBuilder:
    @staticmethod
    def build_lpm_config(
        min_segment_size=5,
        lambda_balance=0,
        context_width=4,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        device=None,
        batch_size=32,
        normalize=True,
    ):
        if device is None:
            device = DeviceResolver.resolve_device()
        return {
            "MIN_SEGMENT_SIZE": min_segment_size,
            "LAMBDA_BALANCE": lambda_balance,
            "UTTERANCE_EXPANSION_WIDTH": context_width,
            "HF_EMBEDDING_MODEL": embedding_model,
            "HF_DEVICE": device,
            "HF_BATCH_SIZE": batch_size,
            "HF_NORMALIZE": normalize,
        }
