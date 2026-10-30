"""Render the README chart from the paper's benchmark summaries."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent


def main():
    data = json.loads((HERE / 'results_summary.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.titleweight': 'bold', 'text.color': '#202a38',
                         'axes.labelcolor': '#4b5563', 'xtick.color': '#6b7280',
                         'ytick.color': '#263244', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout='constrained')
    for ax, setting in zip(axes.flat, data['settings']):
        names = list(setting['methods'])
        scores = [setting['methods'][name]['overall'] for name in names]
        colors = [('#8a2360' if name == 'SpectralKV' else
                   '#48558f' if name == 'SpectralKV-Fixed' else
                   '#b8bec7' if name == 'Full cache' else '#a3bfd9') for name in names]
        bars = ax.barh(names, scores, color=colors, height=0.64, zorder=3)
        ax.invert_yaxis()
        ax.bar_label(bars, labels=[f'{score:.2f}' for score in scores],
                     padding=5, fontsize=10, color='#202a38')
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.grid(axis='x', color='#e6e9ee', linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', length=0, pad=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for tick in ax.get_yticklabels():
            if tick.get_text().startswith('SpectralKV'):
                tick.set_fontweight('bold')
        ax.set_title(f"{setting['model']}  |  {setting['keep_ratio_percent']}% cache", loc='left', pad=14, fontsize=13)
        ax.set_xlabel('Overall score (0–100; higher is better)', labelpad=9, fontsize=10)
    fig.suptitle('SpectralKV at 5% and 10% prompt-cache budgets', fontsize=20, fontweight='bold')
    fig.savefig(HERE / 'results.png', dpi=180, metadata={'Software': 'Matplotlib'})
    plt.close(fig)


if __name__ == '__main__':
    main()
