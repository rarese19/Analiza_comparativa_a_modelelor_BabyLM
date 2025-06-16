import json

with open('LSTM_blimp_bins.json', 'r', encoding='utf-8') as f:
    bins = json.load(f)

for bin_label, items in bins.items():
    print(f"{bin_label}: {len(items)} task-uri")
