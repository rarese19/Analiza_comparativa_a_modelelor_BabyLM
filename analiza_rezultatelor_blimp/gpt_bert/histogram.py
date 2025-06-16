import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import json

def extract_scores_and_tasks(path):
    entries = []
    float_pattern = re.compile(r'([0-9]+\.[0-9]+)')
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('TEMPERATURE:'):
                continue
            if line.upper().startswith('### AVERAGE ACCURACY'):
                break
            if line.startswith('###'):
                continue

            m = float_pattern.search(line)
            if m:
                raw   = float(m.group(1))
                score = raw / 100.0
                task  = line.split(':', 1)[0].strip()
                entries.append((task, score))
    return entries

def collect_all_entries(dir_patterns):
    all_entries = []
    for pat in dir_patterns:
        for fp in glob.glob(pat):
            all_entries.extend(extract_scores_and_tasks(fp))
    return all_entries

def bucket_tasks(entries, bins, labels):
    bin_tasks = {label: [] for label in labels}
    for task, score in entries:
        for i in range(len(bins)-1):
            low, high = bins[i], bins[i+1]
            if (score >= low) and (score < high or (i == len(bins)-2 and score <= high)):
                bin_tasks[labels[i]].append((task, score))
                break
    for label in bin_tasks:
        bin_tasks[label].sort(key=lambda x: x[1])
    return bin_tasks

def plot_histogram(scores, bins, labels, title, save_path):
    max_ylim = 5 if title == "BLiMP Supplement" else 67

    counts, _ = np.histogram(scores, bins=bins)
    x = np.arange(len(labels))

    fig, ax = plt.subplots()
    ax.bar(x, counts, width=0.8, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Intervale de scor')
    ax.set_ylabel('Număr de task-uri')
    ax.set_title(title)

    ax.set_ylim(0, max_ylim)

    yt = list(ax.get_yticks())
    yt[-1] = max_ylim
    ax.set_yticks(yt)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def save_bins_to_json(bin_tasks, filename):
    serializable = {
        label: [{"task": t, "score": s} for t, s in items]
        for label, items in bin_tasks.items()
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"Am salvat bin-urile în '{filename}'")

if __name__ == '__main__':
    dir_patterns = ['blimp/*.txt']

    main_entries = collect_all_entries(dir_patterns)

    dir_patterns = ['blimp_supplement/*.txt']
    supp_entries = collect_all_entries(dir_patterns)

    main_entries = [(t, s) for t, s in main_entries if not t.startswith('supplement_')]
    supp_entries = [(t, s) for t, s in supp_entries if t.startswith('supplement_')]

    bins_main   = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels_main = ['0.00–0.20','0.20–0.40','0.40–0.60','0.60–0.80','0.80–1.00']

    bins_supp, labels_supp = bins_main, labels_main

    bin_main = bucket_tasks(main_entries, bins_main, labels_main)
    bin_supp = bucket_tasks(supp_entries, bins_supp, labels_supp)
    save_bins_to_json(bin_main, 'GPT-BERT_blimp_bins.json')
    save_bins_to_json(bin_supp, 'GPT-BERT_blimp_supplement_bins.json')

    plot_histogram(
        [s for _, s in main_entries],
        bins_main, labels_main,
        title='BLiMP',
        save_path="GPT-BERT_blimp_histogram.pdf"
    )
    plot_histogram(
        [s for _, s in supp_entries],
        bins_supp, labels_supp,
        title='BLiMP Supplement',
        save_path="GPT-BERT_blimp_supplement_histogram.pdf"

    )
