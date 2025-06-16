import json
import sys

file_bad   = "sentential_subject_island.jsonl"
file_preds = "predictions_at_best_temperature.json"

bad_records = []
with open(file_bad, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        bad_records.append(json.loads(line))

if not bad_records:
    print(f"⚠ Nu am găsit niciun record în {file_bad}", file=sys.stderr)
    sys.exit(1)

task_key = bad_records[0].get("UID")
if not task_key:
    print(f"⚠ Primul record nu are câmpul UID!", file=sys.stderr)
    sys.exit(1)

with open(file_preds, encoding="utf-8") as f:
    preds_data = json.load(f)

if task_key not in preds_data:
    print(f"⚠ Task-ul '{task_key}' nu există în {file_preds}", file=sys.stderr)
    sys.exit(1)

preds = preds_data[task_key].get("predictions", [])
pred_map = { p.get("pred","").strip(): p.get("id") for p in preds }

cnt = 0
for rec in bad_records:
    bad = rec.get("sentence_good", "").strip()
    pid = pred_map.get(bad)
    if pid:
        cnt += 1
        print(f"{rec.get('UID')}: ✅ găsit → id = {pid}")

print(cnt)