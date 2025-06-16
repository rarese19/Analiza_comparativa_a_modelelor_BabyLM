import json
import glob
import os

import numpy as np
import matplotlib.pyplot as plt

def load_main_bins_scores(pattern="**/*_blimp_bins.json"):
    model_scores = {}
    for filepath in glob.glob(pattern, recursive=True):
        model = os.path.basename(filepath).split('_blimp')[0]
        with open(filepath, 'r', encoding='utf-8') as f:
            bins = json.load(f)

        scores = [ item["score"]
                   for bucket in bins.values()
                   for item in bucket ]
        model_scores[model] = scores
    return model_scores

def plot_grouped_histogram(model_scores, bins, labels, out_pdf="grouped_histogram.pdf", task='BLiMP'):
    models = list(model_scores.keys())
    n_models = len(models)
    n_bins   = len(bins) - 1
    max_ylim = 67 if task == "BLiMP" else 5

    counts = []
    for m in models:
        cnt, _ = np.histogram(model_scores[m], bins=bins)
        counts.append(cnt)
    counts = np.array(counts)

    x = np.arange(n_bins)
    total_width = 0.8
    width = total_width / n_models

    fig, ax = plt.subplots(figsize=(1.5*n_bins, 4))

    for i, model in enumerate(models):
        ax.bar(
            x + i*width,
            counts[i],
            width=width,
            label=model,
            edgecolor='black'
        )

    ax.set_xticks(x + total_width/2 - width/2)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Intervale de scor')
    ax.set_ylabel('Număr de task-uri')
    ax.legend(title='Model', loc='upper left', frameon=True)

    ax.set_ylim(0, max_ylim)
    yt = list(ax.get_yticks())
    yt[-1] = max_ylim
    ax.set_yticks(yt)

    plt.tight_layout()
    fig.savefig(out_pdf, format='pdf', bbox_inches='tight')
    print(f"→ Histogramă grupată salvată în '{out_pdf}'")
    plt.show()

if __name__ == "__main__":
    model_scores = load_main_bins_scores("**/*_blimp_bins.json")

    bins   = np.linspace(0.0, 1.0, 6)
    labels = ['0.00–0.20','0.20–0.40','0.40–0.60','0.60–0.80','0.80–1.00']

    plot_grouped_histogram(model_scores, bins, labels, out_pdf="all_models_blimp_histogram.pdf", task="BLiMP")

    model_scores = load_main_bins_scores("**/*_blimp_supplement_bins.json")

    plot_grouped_histogram(model_scores, bins, labels, out_pdf="all_models_blimp_supplement_histogram.pdf", task="BLiMP Supplement")


