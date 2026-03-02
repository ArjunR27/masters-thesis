import os
import csv
import bisect
import re

_SLIDE_OCR_RE = re.compile(r"slide_(\d+)_ocr\.csv$", re.IGNORECASE)

def parse_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
    
def parse_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def iter_rows(csv_path):
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row

def extract_ocr_text(csv_path, min_conf=0.0, line_sep="\n"):
    tokens = []
    for row in iter_rows(csv_path):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        conf = parse_float(row.get("conf"), default=None)
        if conf is not None and conf < min_conf:
            continue
        cleaned = " ".join(text.split())
        block = parse_int(row.get("block_num"))
        par = parse_int(row.get("par_num"))
        line = parse_int(row.get("line_num"))
        word = parse_int(row.get("word_num"))
        tokens.append((block, par, line, word, cleaned))

    if not tokens:
        return ""

    tokens.sort()
    lines = []
    current_key = None
    current_words = []
    for block, par, line, word, text in tokens:
        key = (block, par, line)
        if current_key is None:
            current_key = key
        if key != current_key:
            lines.append(" ".join(current_words))
            current_words = [text]
            current_key = key
        else:
            current_words.append(text)
    if current_words:
        lines.append(" ".join(current_words))

    return line_sep.join(lines).strip()


def load_slide_end_times(segments_path):
    end_times = []
    with open(segments_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            value = parse_float(line, default=None)
            if value is None:
                continue
            end_times.append(value)
    return end_times

def load_slide_ocr_texts(slides_dir, min_conf=0.0, line_sep="\n"):
    ocr_by_slide = {}
    for filename in sorted(os.listdir(slides_dir)):
        if not filename.endswith("_ocr.csv"):
            continue
        match = _SLIDE_OCR_RE.search(filename)
        if not match:
            continue
        slide_idx = int(match.group(1))
        path = os.path.join(slides_dir, filename)
        text = extract_ocr_text(path, min_conf=min_conf, line_sep=line_sep)
        if text:
            ocr_by_slide[slide_idx] = text
    return ocr_by_slide

def attach_slide_ocr_to_utterances(
    utterances,
    segments_path,
    slides_dir,
    min_conf=0.0,
    line_sep="\n",
    attach_mode="midpoint",
    attach_ocr_per_slide=1,
):
    end_times = load_slide_end_times(segments_path)
    if not end_times:
        return utterances
    ocr_by_slide = load_slide_ocr_texts(slides_dir, min_conf=min_conf, line_sep=line_sep)

    slide_counts = {}
    for utt in utterances:
        start = utt.get("start")
        end = utt.get("end")
        if start is None or end is None:
            continue
        if attach_mode == "start":
            pivot = start
        elif attach_mode == "end":
            pivot = end
        else:
            pivot = (start + end) / 2.0

        slide_idx = bisect.bisect_left(end_times, pivot)
        if slide_idx >= len(end_times):
            continue

        slide_start = 0.0 if slide_idx == 0 else end_times[slide_idx - 1]
        slide_end = end_times[slide_idx]
        utt["slide_index"] = slide_idx
        utt["slide_start"] = round(slide_start, 3)
        utt["slide_end"] = round(slide_end, 3)

        slide_counts[slide_idx] = slide_counts.get(slide_idx, 0) + 1
        attach_limit = attach_ocr_per_slide
        if attach_limit is None:
            should_attach = True
        elif attach_limit <= 0:
            should_attach = False
        else:
            should_attach = slide_counts[slide_idx] <= attach_limit
        if should_attach:
            ocr_text = ocr_by_slide.get(slide_idx)
            if ocr_text:
                utt["ocr_text"] = ocr_text
    return utterances

def group_words_into_utterances(rows, max_gap_s, lowercase):
    utterances = []
    words_buf = []
    start_time = None
    last_end = None

    def flush():
        nonlocal words_buf, start_time, last_end
        if not words_buf:
            return
        
        text = " ".join(words_buf)
        if lowercase:
            text = text.lower()
        if start_time is not None and last_end is not None:
            utterances.append({
                "start": round(start_time, 3),
                "end": round(last_end, 3),
                "duration": round(last_end - start_time, 3),
                "n_words": len(words_buf),
                "text": text,
            })
        words_buf = []
        start_time = None
        last_end = None

    for row in rows:
        wtext = (row.get("Word") or "").strip()
        if not wtext:
            continue
        try:
            w_start = float(row.get("Start") or 0.0)
        except (ValueError, TypeError):
            w_start = 0.0
        try:
            w_end = float(row.get("End") or w_start)
        except (ValueError, TypeError):
            w_end = w_start
        
        if last_end is not None and (w_start - last_end) >= max_gap_s and words_buf:
            flush()
        
        if start_time is None:
            start_time = w_start
        
        words_buf.append(wtext)
        last_end = w_end
    
    flush()
    return utterances

def extract_utterances_from_transcripts(
    speaker,
    data_dir,
    course_dir,
    max_gap_s,
    lowercase,
    attach_ocr=False,
    ocr_min_conf=0.0,
    ocr_line_sep="\n",
    ocr_per_slide=1,
):
    base = os.path.join(data_dir, speaker, course_dir)
    all_utts = []

    for meeting_id in sorted(os.listdir(base)):
        meet_dir = os.path.join(base, meeting_id)
        transcripts_path = None
        for fn in os.listdir(meet_dir):
            if fn.endswith("_transcripts.csv"):
                transcripts_path = os.path.join(meet_dir, fn)
                break
        if transcripts_path is None:
            continue

        video_id = os.path.basename(transcripts_path).replace("_transcripts.csv", "")
        rows = list(iter_rows(transcripts_path))
        utts = group_words_into_utterances(rows, max_gap_s, lowercase)

        if attach_ocr:
            segments_path = os.path.join(meet_dir, "segments.txt")
            if os.path.exists(segments_path):
                attach_slide_ocr_to_utterances(
                    utts,
                    segments_path=segments_path,
                    slides_dir=meet_dir,
                    min_conf=ocr_min_conf,
                    line_sep=ocr_line_sep,
                    attach_ocr_per_slide=ocr_per_slide,
                )

        for i, u in enumerate(utts):
            u.update({
                "meeting_id": meeting_id,
                "video_id": video_id,
                "idx": i,
                "source": "transcripts",
                "path": transcripts_path,
            })
        all_utts.extend(utts)

    return all_utts


def extract_utterances_from_transcript_file(
    csv_path,
    max_gap_s,
    lowercase,
    segments_path=None,
    slides_dir=None,
    ocr_min_conf=0.0,
    ocr_line_sep="\n",
    ocr_per_slide=1,
):
    rows = list(iter_rows(csv_path))
    utterances = group_words_into_utterances(rows, max_gap_s, lowercase)
    if segments_path and slides_dir and os.path.exists(segments_path):
        attach_slide_ocr_to_utterances(
            utterances,
            segments_path=segments_path,
            slides_dir=slides_dir,
            min_conf=ocr_min_conf,
            line_sep=ocr_line_sep,
            attach_ocr_per_slide=ocr_per_slide,
        )
    return utterances


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "lpm_data")

    ### Extracts the utterances from 1 specified transcript
    utts = extract_utterances_from_transcript_file(
        csv_path="./lpm_data/ml-1/MultimodalMachineLearning/01/VIq5r7mCAyw_transcripts.csv",
        max_gap_s=0.8,
        lowercase=True,
        segments_path="./lpm_data/ml-1/MultimodalMachineLearning/01/segments.txt",
        slides_dir="./lpm_data/ml-1/MultimodalMachineLearning/01",
        ocr_min_conf=60.0,
    )



    
