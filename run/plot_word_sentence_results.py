"""
plot_word_sentence_results.py
────────────────────────────────────────────────────────────────────────────
Plots CER and exact-match accuracy for word-level vs sentence-level
recognition, with and without language-model post-processing, as a
2-panel grouped bar figure suitable for a paper.

Usage:
    python plot_word_sentence_results.py
    python plot_word_sentence_results.py --save results_figure.png
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

# Results — edit these to update the figure.
RESULTS = {
    'Word\n(n=200)': {
        'CER':      {'No LM': 11.9, 'With LM': 23.2},
        'Accuracy': {'No LM': 61.5, 'With LM': 65.0},
    },
    'Sentence\n(n=62)': {
        'CER':      {'No LM': 12.5, 'With LM': 8.5},
        'Accuracy': {'No LM': 24.2, 'With LM': 90.3},
    },
}

COLOR_NO_LM = '#888780'    # c-gray 400 — neutral, "raw model output"
COLOR_LM    = '#378ADD'    # c-blue 400 — "with post-processing"


def plot_results(save_path: str | None):
    conditions = list(RESULTS.keys())
    metrics = ['CER', 'Accuracy']
    ylabels = ['Character Error Rate (%)', 'Exact-match accuracy (%)']

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    bar_width = 0.32
    x = np.arange(len(conditions))

    for ax, metric, ylabel in zip(axes, metrics, ylabels):
        no_lm_vals = [RESULTS[c][metric]['No LM'] for c in conditions]
        lm_vals    = [RESULTS[c][metric]['With LM'] for c in conditions]

        bars1 = ax.bar(x - bar_width / 2, no_lm_vals, bar_width, label='No LM', color=COLOR_NO_LM)
        bars2 = ax.bar(x + bar_width / 2, lm_vals, bar_width, label='With LM', color=COLOR_LM)

        for bars in (bars1, bars2):
            for b in bars:
                height = b.get_height()
                ax.annotate(f'{height:.1f}', xy=(b.get_x() + b.get_width() / 2, height),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(no_lm_vals + lm_vals) * 1.2)
        ax.set_title(metric if metric == 'CER' else 'Exact-match accuracy')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[1].legend(loc='upper left', frameon=False)
    fig.suptitle('Word- vs. sentence-level recognition, with/without LM post-processing')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'[saved] {save_path}')
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot word/sentence CER + accuracy results')
    parser.add_argument('--save', default=None, help='Save the figure to this path instead of showing it')
    args = parser.parse_args()
    plot_results(args.save)


if __name__ == '__main__':
    main()