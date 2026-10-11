"""Simple interactive Q&A app for LPM lecture videos.

Lets you pick a lecture (from LPM_QA_DATASET), ask a question, and choose
which of the 14 retrieval configurations from ragas_evaluation/ragas_eval.py
(2 TreeSeg index kinds + 12 baseline chunking configs) answers it. Reuses
that script's store-building, retrieval, and context-formatting logic
directly instead of reimplementing it; only the synchronous OpenAI answer
call and the Streamlit UI/caching are new.

Run with: streamlit run qa_app.py
"""

from __future__ import annotations

import csv
import html
import json
import sys
from pathlib import Path

import streamlit as st
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ragas_evaluation.ragas_eval as ragas_eval

SYSTEMS_BY_NAME = {system["name"]: system for system in ragas_eval.SYSTEMS}
CUSTOM_QUESTION_OPTION = "— type your own question —"


def load_dataset_rows(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            question = (row.get("question") or "").strip()
            lecture_key = (row.get("lecture_key") or "").strip()
            answer_text = (row.get("answer_text") or "").strip()
            if not question or not lecture_key or not answer_text:
                continue
            rows.append(
                {
                    "question": question,
                    "lecture_key": lecture_key,
                    "answer_text": answer_text,
                    "answer_timestamps": (row.get("answer_timestamps") or "").strip(),
                    "question_type": (row.get("question_type") or "").strip(),
                }
            )
    return rows


@st.cache_resource
def get_dataset_rows():
    return load_dataset_rows(ragas_eval.DATASET_PATH)


@st.cache_resource
def get_lectures():
    return ragas_eval.discover_lectures(ragas_eval.LPM_DATA_DIR)


@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=ragas_eval.OPENAI_API_KEY)


@st.cache_resource
def get_asr_store(system_name: str, lecture_key: str):
    system = SYSTEMS_BY_NAME[system_name]
    lecture = get_lectures()[lecture_key]
    return ragas_eval.build_store(system, [lecture], openai_client=get_openai_client())


@st.cache_resource
def get_ocr_store(lecture_key: str):
    lecture = get_lectures()[lecture_key]
    return ragas_eval.VectorStoreFactory().build_ocr_vector_store(
        [lecture], embed_model=ragas_eval.EMBEDDING_MODEL, normalize=True, build_global=False,
    )


@st.cache_resource
def get_reranker(kind: str):
    return ragas_eval.build_reranker(kind)


@st.cache_resource
def get_ocr_reranker():
    return ragas_eval.build_ocr_reranker()


def _hit_intervals(hits: list[dict], kind: str) -> list[tuple[float, float, str]]:
    """(start, end, text) tuples to mark, using leaf-level precision for summary-tree nodes."""
    intervals = []
    for hit in hits:
        if kind == "summary_tree" and not hit.get("is_leaf", True):
            for leaf in hit.get("supporting_leaves") or []:
                if leaf.get("start") is not None and leaf.get("end") is not None:
                    intervals.append((float(leaf["start"]), float(leaf["end"]), str(leaf.get("text") or "")))
            continue
        if hit.get("start") is not None and hit.get("end") is not None:
            intervals.append((float(hit["start"]), float(hit["end"]), str(hit.get("text") or "")))
    return intervals


def _merge_intervals(intervals: list[tuple[float, float, str]]) -> list[dict]:
    """Merge overlapping/adjacent (start, end, text) intervals into display segments."""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda iv: iv[0])
    merged = [{"start": ordered[0][0], "end": ordered[0][1], "label": ordered[0][2]}]
    for start, end, text in ordered[1:]:
        last = merged[-1]
        if start <= last["end"] + 0.01:
            last["end"] = max(last["end"], end)
        else:
            merged.append({"start": start, "end": end, "label": text})
    return merged


def build_markers(kind: str, asr_hits: list[dict], ocr_hits: list[dict]) -> dict:
    """Two independent marker tracks (ASR, OCR). Rendered as two stacked rows so overlap
    is shown by vertical alignment instead of a third blended color."""
    asr_intervals = _hit_intervals(asr_hits, kind)
    ocr_intervals = [
        (float(h["start"]), float(h["end"]), str(h.get("text") or ""))
        for h in ocr_hits
        if h.get("start") is not None and h.get("end") is not None
    ]
    return {"asr": _merge_intervals(asr_intervals), "ocr": _merge_intervals(ocr_intervals)}


ASR_COLOR = "#e04040"
OCR_COLOR = "#2fa84f"


def render_video_with_markers(video_id: str, markers: dict) -> None:
    markers_for_js = {
        track: [{**m, "label": html.escape(m["label"])} for m in segs]
        for track, segs in markers.items()
    }
    component_html = f"""
    <div style="max-width: 960px; font-family: sans-serif;">
      <div style="position:relative; padding-top:56.25%; background:#000; border-radius:4px; overflow:hidden;">
        <div id="yt-player" style="position:absolute; top:0; left:0; width:100%; height:100%;"></div>
      </div>
      <div style="display:flex; align-items:center; gap:6px; margin-top:8px;">
        <span style="width:60px; font-size:12px; color:{ASR_COLOR}; font-weight:bold;">ASR</span>
        <div id="timeline-asr" style="position:relative; flex:1; height:20px; background:#e5e5e5;
                                       border-radius:4px; overflow:hidden;"></div>
      </div>
      <div style="display:flex; align-items:center; gap:6px; margin-top:4px;">
        <span style="width:60px; font-size:12px; color:{OCR_COLOR}; font-weight:bold;">OCR</span>
        <div id="timeline-ocr" style="position:relative; flex:1; height:20px; background:#e5e5e5;
                                       border-radius:4px; overflow:hidden;"></div>
      </div>
      <div style="margin-top:6px; font-size:12px; color:#666;">
        <span style="display:inline-block;width:10px;height:10px;background:{ASR_COLOR};margin-right:4px;"></span>Transcript (ASR)
        <span style="display:inline-block;width:10px;height:10px;background:{OCR_COLOR};margin:0 4px 0 14px;"></span>Slide (OCR)
        — markers lined up in both rows mean both sources cover that moment.
      </div>
    </div>
    <script>
      (function() {{
        const markers = {json.dumps(markers_for_js)};
        const videoId = {json.dumps(video_id)};

        function drawTrack(elementId, segments, color, duration) {{
          const timeline = document.getElementById(elementId);
          if (!timeline || !duration) return;
          timeline.innerHTML = "";
          segments.forEach(function(m) {{
            const div = document.createElement("div");
            const left = Math.max(0, Math.min(100, (m.start / duration) * 100));
            const width = Math.max(0.3, Math.min(100 - left, ((m.end - m.start) / duration) * 100));
            div.style.position = "absolute";
            div.style.top = "0";
            div.style.height = "100%";
            div.style.left = left + "%";
            div.style.width = width + "%";
            div.style.background = color;
            div.style.cursor = "pointer";
            div.title = m.label + " (" + m.start.toFixed(0) + "s-" + m.end.toFixed(0) + "s)";
            div.addEventListener("click", function() {{
              if (window.ytPlayer) {{
                window.ytPlayer.seekTo(m.start, true);
                window.ytPlayer.playVideo();
              }}
            }});
            timeline.appendChild(div);
          }});
        }}

        function drawMarkers(duration) {{
          drawTrack("timeline-asr", markers.asr, {json.dumps(ASR_COLOR)}, duration);
          drawTrack("timeline-ocr", markers.ocr, {json.dumps(OCR_COLOR)}, duration);
        }}

        function createPlayer() {{
          window.ytPlayer = new YT.Player("yt-player", {{
            videoId: videoId,
            events: {{
              onReady: function(e) {{ drawMarkers(e.target.getDuration()); }}
            }}
          }});
        }}

        if (window.YT && window.YT.Player) {{
          createPlayer();
        }} else {{
          const tag = document.createElement("script");
          tag.src = "https://www.youtube.com/iframe_api";
          document.body.appendChild(tag);
          window.onYouTubeIframeAPIReady = createPlayer;
        }}
      }})();
    </script>
    """
    st.iframe(component_html, height=600)


def answer_question(system_name: str, lecture_key: str, question: str) -> dict:
    system = SYSTEMS_BY_NAME[system_name]
    kind = system["kind"]

    # Two fully independent indexes, searched separately — same as ragas_eval.py's run_retrieval.
    asr_store = get_asr_store(system_name, lecture_key)
    ocr_store = get_ocr_store(lecture_key)

    asr_hits = ragas_eval.retrieve_hits(system, asr_store, question, lecture_key, get_reranker(kind))
    asr_context = ragas_eval.build_context(asr_hits, kind)

    ocr_hits: list[dict] = []
    if lecture_key in ocr_store.lecture_indices:
        raw_ocr = ocr_store.search(question, top_k=ragas_eval.OCR_TOP_K, lecture_key=lecture_key)
        ocr_hits = get_ocr_reranker().rerank(question, raw_ocr, top_n=ragas_eval.OCR_TOP_N)
    ocr_context = ragas_eval.build_ocr_context(ocr_hits)
    markers = build_markers(kind, asr_hits, ocr_hits)

    if asr_context and ocr_context:
        context = asr_context + "\n\n--- Slide Evidence ---\n\n" + ocr_context
    else:
        context = asr_context or ocr_context

    if not context:
        answer = ragas_eval.INSUFFICIENT_CONTEXT_RESPONSE
    else:
        user_content = (
            f"Retrieved lecture segments:\n{'─' * 60}\n{context}\n{'─' * 60}\n\n"
            f"Student question: {question}"
        )
        resp = get_openai_client().chat.completions.create(
            model=ragas_eval.GENERATOR_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": ragas_eval.QUERY_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
        )
        answer = resp.choices[0].message.content.strip()

    return {"answer": answer, "asr_context": asr_context, "ocr_context": ocr_context, "markers": markers}


st.set_page_config(page_title="Multimodal Lecture Q&A", layout="wide")
st.title("LPM Lecture Q&A")

if not ragas_eval.OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set in .env — answer generation will fail.")

dataset_rows = get_dataset_rows()
lectures_by_key = get_lectures()
usable_rows = [row for row in dataset_rows if row["lecture_key"] in lectures_by_key]

if not usable_rows:
    st.error("No usable questions found — check LPM_QA_DATASET/lpm_qa_labeled.csv and lpm_data/.")
    st.stop()

lecture_keys = sorted({row["lecture_key"] for row in usable_rows})
lecture_key = st.selectbox(
    "Lecture", lecture_keys, format_func=lambda k: lectures_by_key[k].label,
)
video_placeholder = st.empty()

lecture_rows = [row for row in usable_rows if row["lecture_key"] == lecture_key]
question_choice = st.selectbox(
    "Sample question (optional)", [CUSTOM_QUESTION_OPTION] + [row["question"] for row in lecture_rows],
)

selected_row = None
if question_choice != CUSTOM_QUESTION_OPTION:
    selected_row = next(row for row in lecture_rows if row["question"] == question_choice)

question = st.text_area(
    "Question", value=selected_row["question"] if selected_row else "", height=80,
)

system_names = [system["name"] for system in ragas_eval.SYSTEMS]
system_name = st.selectbox("Configuration", system_names, index=system_names.index("treeseg_leaf"))

markers: dict = {"asr": [], "ocr": []}
if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner(f"Running {system_name} on {lecture_key}..."):
            result = answer_question(system_name, lecture_key, question.strip())
        markers = result["markers"]

        st.subheader("Answer")
        st.write(result["answer"])

        with st.expander("Retrieved transcript / tree evidence"):
            st.text(result["asr_context"] or "(no transcript evidence retrieved)")

        with st.expander("Retrieved slide (OCR) evidence"):
            st.text(result["ocr_context"] or "(no slide evidence retrieved)")

        if selected_row is not None:
            with st.expander("Reference answer (gold, from LPM_QA_DATASET)"):
                st.write(f"**Answer:** {selected_row['answer_text']}")
                if selected_row["question_type"]:
                    st.write(f"**Question type:** {selected_row['question_type']}")
                if selected_row["answer_timestamps"]:
                    st.write(f"**Timestamps:** {selected_row['answer_timestamps']}")

with video_placeholder.container():
    render_video_with_markers(lectures_by_key[lecture_key].video_id, markers)
