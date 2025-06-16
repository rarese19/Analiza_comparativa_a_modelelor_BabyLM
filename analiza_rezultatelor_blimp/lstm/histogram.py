import json
import glob
import numpy as np
import matplotlib.pyplot as plt

def collect_entries_from_json(patterns, exclude):
    entries = []
    for pat in patterns:
        for fp in glob.glob(pat):
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            results = data.get('results', {})
            for raw_task, vals in results.items():
                if raw_task in exclude:
                    continue
                acc = vals.get('acc,none')
                if acc is None:
                    continue

                task = raw_task
                if task.startswith("blimp_"):
                    task = task[len("blimp_"):]
                if task.endswith("_filtered"):
                    task = task[:-len("_filtered")]

                entries.append((task, float(acc)))
    return entries

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

def print_bins(bin_tasks):
    print("\nDistribuția task-urilor pe intervale:\n")
    for label, items in bin_tasks.items():
        print(f"{label}:")
        if items:
            for task, score in items:
                print(f"  - {task}: {score:.3f}")
        else:
            print("  (niciun task)")
        print()

def save_bins_to_json(bin_tasks, filename):
    serializable = {
        label: [{"task": t, "score": s} for t, s in items]
        for label, items in bin_tasks.items()
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"Am salvat bin-urile în '{filename}'")

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

if __name__ == '__main__':
    json_patterns = ['*.json']

    exclude = ['blimp_filtered', 'blimp_supplement']

    entries = collect_entries_from_json(json_patterns, exclude)

    main_entries = [(t, s) for t, s in entries if not t.startswith('supplement_')]
    supp_entries = [(t, s) for t, s in entries if t.startswith('supplement_')]

    bins_main   = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels_main = ['0.00–0.20','0.20–0.40','0.40–0.60','0.60–0.80','0.80–1.00']

    bin_main = bucket_tasks(main_entries, bins_main, labels_main)
    print_bins(bin_main)
    save_bins_to_json(bin_main, 'LSTM_blimp_bins.json')
    plot_histogram(
        [s for _, s in main_entries],
        bins_main, labels_main,
        title='BLiMP',
        save_path="LSTM_blimp_histogram.pdf"

    )

    bin_supp = bucket_tasks(supp_entries, bins_main, labels_main)
    print_bins(bin_supp)
    save_bins_to_json(bin_supp, 'LSTM_blimp_supplement_bins.json')
    plot_histogram(
        [s for _, s in supp_entries],
        bins_main, labels_main,
        title='BLiMP Supplement',
        save_path = "LSTM_blimp_supplement_histogram.pdf"
    )
