"""Plot actual results only; no hard-coded accuracy values."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.input / 'summary.json').read_text(encoding='utf-8'))
    env = json.loads((args.input / 'environment.json').read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = dict(full='#222222', uniform='#777777', staged='#2673c9', linear='#20a68a', late='#df8630')
    for family, color in colors.items():
        rows = sorted([r for r in summary['rows'] if r['name'].split('_')[0] == family], key=lambda r: r['avg_gmacs'])
        ax.plot([r['avg_gmacs'] for r in rows], [r['top1'] for r in rows], 'o-', color=color, label=family)
        for r in rows:
            if family != 'uniform':
                ax.annotate(r['name'], (r['avg_gmacs'], r['top1']), fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax.set(xlabel='Analytic average GMACs / image', ylabel='Top-1 (%)',
           title=f"Layer-wise drop | n={env['arguments']['samples']} | threshold={summary['threshold']}")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.output / 'pareto.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    for r in summary['rows']:
        if r['name'] in ('full', 'uniform_50', 'staged_50', 'linear_50', 'late_50'):
            ax.step(range(1, 13), r['counts'], where='mid', label=r['name'])
    ax.set(xlabel='6H block (1-based; count on entry)', ylabel='Tokens including CLS', xticks=range(1, 13))
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.output / 'schedules.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
