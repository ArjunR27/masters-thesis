from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import yake

from treeseg_lpm import (
    load_lpm_utterances,
    build_lpm_config,
    build_treeseg_entries
)

from treeseg import TreeSeg

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "lpm_data"


YAKE_EXTRACTOR = yake.KeywordExtractor(
    lan='en',
    n=3,
    top=5,
    dedupLim=0.9,
    windowsSize=1,
    features=None
)


def collect_segment_embeddings(csv_path, speaker='ml-1', course_dir='MultimodalMachineLearning', meeting_id='',
                               min_segment_size=5, lambda_balance=0, context_width=4, target_segments=None):
    
    utterances = load_lpm_utterances(csv_path=csv_path, data_dir=DATA_DIR, speaker=speaker, course_dir=course_dir,
                                     meeting_id=meeting_id, max_gap_s=0.8, lowercase=True)
    
    entries = build_treeseg_entries(utterances)
    configs = build_lpm_config(min_segment_size=min_segment_size, lambda_balance=lambda_balance, context_width=context_width)

    model = TreeSeg(configs=configs, entries=list(entries))

    if target_segments is None:
        model.segment_meeting(K=float('inf'))
    else:
        model.segment_meeting(K=target_segments)
    
    # block_embeddings is an np.ndarray [num_blocks, dim]
    ## the block as described in the paper is the [-x_amount of utt + curr_utt + x_amount of utt]
    block_embeddings = np.vstack([block["embedding"] for block in model.blocks])

    segments = []
    for seg_idx, leaf in enumerate(model.leaves, start=1):
        indices = leaf.segment
        segment_utts = [entries[i] for i in indices]
        segments.append(
            {
                "segment_id": seg_idx,
                "tree_path": leaf.identifier,
                "utterance_start": indices[0],
                "utterance_end": indices[-1],
                "n_utterances": len(indices),
                'start': segment_utts[0].get("start"),
                'end': segment_utts[-1].get("end"),
                "text": "\n".join(utt.get("composite", "") for utt in segment_utts).strip(),
                "block_indices": indices,
            }
        )
    
    return block_embeddings, segments

# Takes the corresponds blocks for a segment, averages those vectors into a --> segment embedding vector
# this way we can actuall compare vectors for each segment rather than the blocks that make up the segmnets
def convert_block_embedding_to_segment_embedding(block_embeddings, segments):
    segment_vectors = []
    for segment in segments:
        idxs = segment['block_indices']
        seg_matrix = block_embeddings[idxs]
        seg_vec = seg_matrix.mean(axis=0)
        seg_vec /= np.linalg.norm(seg_vec) + 1e-12
        segment_vectors.append(seg_vec)
    
    return segment_vectors

# normalizes to a unit vector we just want to compare the cosine/directions
def normalize(vecs, eps=1e-12):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / (norms + eps)

def comp_adj_segments(segment_vectors):
    # cosine similarity between adjacent vectors 
    ## this is to determine if there are clear topic shifts?
    ### in the next phase i would want to know which vectors are similar because they may tlak about the same subject
    adjacent_cosine = []
    for i in range(1, len(segment_vectors)):
        left = segment_vectors[i-1]
        right = segment_vectors[i]
        cosine = float((left * right).sum())
        adjacent_cosine.append(cosine)
    
    return adjacent_cosine

def extract_segment_keywords(segment_text, top_k=5):
    scored = YAKE_EXTRACTOR.extract_keywords(segment_text)
    keywords = [kw for kw, score in scored[:top_k]]
    return keywords

# plots the segment embeddings on a 2d plane utilizing PCA to reduce the dimensionality
## we can visualize which segments are close to one another on a 2d-plane indicating which segments may be near one another
def plot_embeddings(segments, segment_vecs):
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(segment_vecs)

    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c=np.arange(len(coords)), cmap="viridis")
    for i, (x, y) in enumerate(coords):
        plt.text(x + 0.01, y + 0.01, str(i + 1), fontsize=8)

    plt.title("Segment embeddings projected to 2-D (color = time order)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.colorbar(label="Segment index")
    plt.tight_layout()
    plt.show()


def main():
    # first video in the ML1 lecture
    csv_path2 = "/Users/arjunranade/CP_CSC/MastersThesis/masters-thesis/lpm_data/ml-1/MultimodalMachineLearning/02/fBYu8I52nVM_transcripts.csv"
    csv_path3 = "/Users/arjunranade/CP_CSC/MastersThesis/masters-thesis/lpm_data/ml-1/MultimodalMachineLearning/08/2_dZ5GBlRgU_transcripts.csv"
    csv_path="./lpm_data/ml-1/MultimodalMachineLearning/01/VIq5r7mCAyw_transcripts.csv"
    embeddings, segments = collect_segment_embeddings(csv_path3)

    segment_vectors = convert_block_embedding_to_segment_embedding(embeddings, segments)
    segment_vectors = normalize(np.vstack(segment_vectors))
    adjacent_cosine_scores = comp_adj_segments(segment_vectors)

    
    boundary_info = []
    for idx, cosine in enumerate(adjacent_cosine_scores):
        boundary_info.append(
            {
                "boundary_id": idx+1,
                "cosine": cosine,
                "left_segment": segments[idx]["segment_id"],
                "right_segment": segments[idx + 1]["segment_id"],
                "left_text": segments[idx]["text"],
                "right_text": segments[idx + 1]["text"]
            }
        )
    
    # print(len(boundary_info))
    # boundary = boundary_info[20]
    # print(boundary["left_segment"], boundary['left_text'])
    # print(boundary["right_segment"], boundary["right_text"])

    # keywords_left = extract_segment_keywords(boundary["left_text"])
    # keywords_right = extract_segment_keywords(boundary["right_text"])

    # print(keywords_left)
    # print(keywords_right)

    for idx, seg in enumerate(segments):
        keywords = extract_segment_keywords(seg['text'])
        print(f"{idx}: {keywords}")

    # printing how many utteracnes/blocks are in each segment
    # for seg in segments:
    #     print(seg["n_utterances"])
    plot_embeddings(segments, segment_vecs=segment_vectors)



if __name__ == "__main__":
    main()
