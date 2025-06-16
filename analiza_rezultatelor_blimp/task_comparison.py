import json
import pandas as pd
import os

model_files = {
    'Frecvența Cuvintelor': os.path.join('frecventa', 'Frecvența Cuvintelor_blimp_bins.json'),
    'GPT-BERT': os.path.join('gpt_bert', 'GPT-BERT_blimp_bins.json'),
    'KenLM':    os.path.join('kenlm',    'KenLM_blimp_bins.json'),
    'LSTM':     os.path.join('lstm',     'LSTM_blimp_bins.json'),
}

chosen_model = 'GPT-BERT'

def load_model_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    for bucket in data.values():
        for entry in bucket:
            rows.append({
                'task':  entry['task'],
                'score': entry['score']
            })
    df = pd.DataFrame(rows).drop_duplicates(subset=['task']).set_index('task')
    return df

dfs = {}
for name, path in model_files.items():
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Couldn't find JSON for model {name!r} at {path!r}")
    dfs[name] = load_model_json(path)

if chosen_model not in dfs:
    raise ValueError(f"Unknown model {chosen_model!r}; available: {list(dfs.keys())}")

chosen_df = dfs[chosen_model]
top3    = chosen_df.nlargest(3, 'score')
bottom3 = chosen_df.nsmallest(3, 'score')

print(f"\nTop 3 tasks for '{chosen_model}':")
print(top3, '\n')

print(f"Bottom 3 tasks for '{chosen_model}':")
print(bottom3, '\n')

compare_tasks = list(top3.index) + list(bottom3.index)
comparison_df = pd.DataFrame(index=compare_tasks)

for name, df in dfs.items():
    comparison_df[name] = df['score']

print("Comparison across all models (for those 10 tasks):")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(comparison_df, '\n')

out_csv = f"comparison_{chosen_model}.csv"
comparison_df.to_csv(out_csv)
print(f"Saved full comparison to {out_csv}")
