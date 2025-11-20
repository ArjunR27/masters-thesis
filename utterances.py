import os
import csv

def iter_transcript_rows(csv_path):
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row

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

def extract_utterances_from_transcripts(speaker, data_dir, course_dir, max_gap_s, lowercase):
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
        rows = list(iter_transcript_rows(transcripts_path))
        utts = group_words_into_utterances(rows, max_gap_s, lowercase)
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


def extract_utterances_from_transcript_file(csv_path, max_gap_s, lowercase):
    rows = list(iter_transcript_rows(csv_path))
    return group_words_into_utterances(rows, max_gap_s, lowercase)

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "lpm_data")
    
    ### Extracts all utterances from the transcripts within a 'speaker' or subject
    # utts = extract_utterances_from_transcripts(
    #     speaker="ml-1",
    #     data_dir=data_dir,
    #     course_dir='MultimodalMachineLearning',
    #     max_gap_s=0.8,
    #     lowercase=True,
    # )

    ### Extracts the utterances from 1 specified transcript
    utts = extract_utterances_from_transcript_file(
        csv_path="./lpm_data/ml-1/MultimodalMachineLearning/01/VIq5r7mCAyw_transcripts.csv",
        max_gap_s=0.8,
        lowercase=True
    )
    print(f"Extracted {len(utts)} utterances (transcripts)")

    print(utts[3])


    