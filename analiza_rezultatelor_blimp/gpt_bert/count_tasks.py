import json

with open('GPT-BERT_blimp_bins.json', 'r', encoding='utf-8') as f:
    bins = json.load(f)

cnt = 0
for bin_label, items in bins.items():
    cnt += len(items)
    print(f"{bin_label}: {len(items)} task-uri")
print(cnt)