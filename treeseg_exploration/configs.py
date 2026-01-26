# Replace OpenAI headers/endpoint with local HF model config
import os
import torch
if torch.cuda.is_available():
    HF_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    HF_DEVICE = "mps"
else:
    HF_DEVICE = "cpu"

# HF_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 80MB, very fast
HF_BATCH_SIZE = 32
HF_NORMALIZE = True

treeseg_configs = {
    "ami": {
        "MIN_SEGMENT_SIZE": 5,
        "LAMBDA_BALANCE": 0,
        "UTTERANCE_EXPANSION_WIDTH": 4,
        "HF_EMBEDDING_MODEL": HF_EMBEDDING_MODEL,
        "HF_DEVICE": HF_DEVICE,
        "HF_BATCH_SIZE": HF_BATCH_SIZE,
        "HF_NORMALIZE": HF_NORMALIZE,
    },
    "icsi": {
        "MIN_SEGMENT_SIZE": 5,
        "LAMBDA_BALANCE": 0,
        "UTTERANCE_EXPANSION_WIDTH": 4,
        "HF_EMBEDDING_MODEL": HF_EMBEDDING_MODEL,
        "HF_DEVICE": HF_DEVICE,
        "HF_BATCH_SIZE": HF_BATCH_SIZE,
        "HF_NORMALIZE": HF_NORMALIZE,
    },
    "augmend": {
        "MIN_SEGMENT_SIZE": 5,
        "LAMBDA_BALANCE": 0,
        "UTTERANCE_EXPANSION_WIDTH": 2,
        "HF_EMBEDDING_MODEL": HF_EMBEDDING_MODEL,
        "HF_DEVICE": HF_DEVICE,
        "HF_BATCH_SIZE": HF_BATCH_SIZE,
        "HF_NORMALIZE": HF_NORMALIZE,
    },
}