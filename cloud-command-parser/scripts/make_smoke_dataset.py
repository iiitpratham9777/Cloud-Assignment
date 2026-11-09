"""Create a tiny SYNTHETIC dataset to test that the training code runs (no Ollama needed).

  python scripts/make_smoke_dataset.py [--count 64] [--out data/smoke_train.jsonl]
  python -m training.train --subset 32 --train-file data/smoke_train.jsonl

The records follow the same format as the cleaned teacher dataset (id, text, teacher_raw) but
the labels come from fixed templates, NOT from the teacher model. A model trained on this data
is only useful to prove the pipeline works; do not report its quality as a result. The output
file is git-ignored.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from service.settings import resolve_path  # noqa: E402

DATES = ["today", "tomorrow", "saturday", "sunday", "monday", "the month of may", "next friday"]
TIMES = ["seven am", "six thirty pm", "five in the morning", "noon", "after five hours", "nine pm"]
PLACES = ["london", "tokyo", "paris", "new york", "los angeles", "berlin", "mumbai"]
EVENTS = ["running", "dentist", "lunch with sam", "team meeting"]

# (intent, text template, slot names used). Templates only contain slot values verbatim.
TEMPLATES = [
    ("alarm_set", "set an alarm for {time} {date}", ("date", "time")),
    ("alarm_set", "wake me up at {time}", ("time",)),
    ("alarm_query", "what alarms do i have set for {date}", ("date",)),
    ("calendar_set", "add {event} to my calendar for {date}", ("date",)),
    ("calendar_query", "what are my plans for {date}", ("date",)),
    ("calendar_remove", "delete my appointment {date}", ("date",)),
    ("weather_query", "what is the weather in {place}", ("place",)),
    ("weather_query", "will it rain in {place} {date}", ("date", "place")),
    ("datetime_query", "what time is it in {place}", ("place",)),
    ("recommendation_events", "what events are happening in {place} {date}", ("date", "place")),
]
POOLS = {"date": DATES, "time": TIMES, "place": PLACES}


def build(count):
    records, seen, i = [], set(), 0
    while len(records) < count and i < count * 50:
        intent, template, slots = TEMPLATES[i % len(TEMPLATES)]
        values = {s: POOLS[s][(i // len(TEMPLATES) + 3 * k) % len(POOLS[s])] for k, s in enumerate(slots)}
        text = template.format(event=EVENTS[i % len(EVENTS)], **{**{"date": "", "time": "", "place": ""},
                                                                  **values})
        text = " ".join(text.split())
        i += 1
        if text in seen:
            continue
        seen.add(text)
        frame = {"intent": intent, "date": None, "time": None, "place": None}
        frame.update({s: v for s, v in values.items()})
        records.append({"id": f"S{len(records) + 1:04d}", "text": text,
                        "teacher_raw": json.dumps(frame, ensure_ascii=False)})
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=64)
    ap.add_argument("--out", default="data/smoke_train.jsonl")
    a = ap.parse_args()
    records = build(a.count)
    out = resolve_path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} SYNTHETIC records to {out} (for pipeline smoke tests only)")


if __name__ == "__main__":
    main()
