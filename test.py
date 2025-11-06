import os
import json
import sys

def extract_utterances(speaker, data_dir, source, max_gap_s, lowercase=True):
    base_dir = os.path.join(data_dir, speaker)
    if source == "figs":
        json_path = os.path.join(base_dir, f"{speaker}_figs.json")
    elif source == "slides":
        json_path = os.path.join(base_dir, f"{speaker}.json")
    
    with open(json_path, "r") as f:
        data = json.load(f)

    utterances = []

    for img_id, item in data.items():
        caps = item.get("captions", [])
        if not caps:
            continue
    
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
                    "img_id": img_id,
                    "scene_id": item.get("scene_id"),
                    "source": source,
                    "start": float(round(start_time, 3)),
                    "end": float(round(last_end, 3)),
                    "duration": float(round(last_end - start_time, 3)),
                    "n_words": len(words_buf),
                    "text": text,
                })
            words_buf = []
            start_time = None
            last_end = None
        
        for w in caps:
            wtext = str(w.get("Word", "")).strip()
            if not wtext:
                continue
            w_start = float(w.get("Start", 0.0) or 0.0)
            w_end = float(w.get("End", w_start) or w_start)

            if last_end is not None and (w_start - last_end) > max_gap_s and words_buf:
                flush()
            
            if start_time is None:
                start_time = w_start
            
            words_buf.append(wtext)
            last_end = w_end
    return utterances

def main():
    utts = extract_utterances("ml-1", data_dir="./lpm_data", source="figs", max_gap_s=0.8)
    print(len(utts), "utterances")
    print(utts[0])
    sys.stdout.flush()


if __name__ == "__main__":
    main()
