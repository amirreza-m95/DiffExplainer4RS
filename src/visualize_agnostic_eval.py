import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_eval_results(pkl_path: str) -> List[Dict[str, Any]]:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("Unexpected eval results format; expected non-empty list of per-user dicts")
    return data


def aggregate_results(per_user: List[Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
    # Determine number of steps (bins)
    max_steps = max(len(u.get('bins', [])) for u in per_user)

    # Metrics available in eval_embedding_diffusion_agnostic.py
    pos_keys = [1, 5, 10, 20, 50, 100]
    neg_keys = [1, 5, 10, 20, 50, 100]

    # Accumulators per step
    acc = {step: {
        'DEL': [], 'INS': [], 'NDCG': [],
        **{f'POS@{k}': [] for k in pos_keys},
        **{f'NEG@{k}': [] for k in neg_keys},
    } for step in range(1, max_steps + 1)}

    for u in per_user:
        del_series = u.get('del', [])
        ins_series = u.get('ins', [])
        ndcg_series = u.get('ndcg', [])
        posk = u.get('pos@k', {})
        negk = u.get('neg@k', {})

        steps = min(len(del_series), len(ins_series), len(ndcg_series), max_steps)
        for s in range(1, steps + 1):
            acc[s]['DEL'].append(del_series[s-1])
            acc[s]['INS'].append(ins_series[s-1])
            acc[s]['NDCG'].append(ndcg_series[s-1])

            for k in pos_keys:
                seq = posk.get(k, [])
                if len(seq) >= s:
                    acc[s][f'POS@{k}'].append(seq[s-1])
            for k in neg_keys:
                seq = negk.get(k, [])
                if len(seq) >= s:
                    acc[s][f'NEG@{k}'].append(seq[s-1])

    # Convert to means per step
    out: Dict[int, Dict[str, float]] = {}
    for s, metrics in acc.items():
        out[s] = {}
        for m, vals in metrics.items():
            if len(vals):
                out[s][m] = float(np.mean(vals))
    return out


def plot_line_groups(agg: Dict[int, Dict[str, float]], output_dir: Path):
    steps = sorted(agg.keys())

    def series(keys: List[str]):
        return {k: [agg[s].get(k, np.nan) for s in steps] for k in keys}

    # POS subset commonly used in papers
    pos_subset = ['POS@1', 'POS@5', 'POS@10']
    pos_data = series(pos_subset)
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, v in pos_data.items():
        ax.plot(steps, v, marker='o', linewidth=2, label=k)
    ax.set_xlabel('Removal Steps (bins)')
    ax.set_ylabel('Score')
    ax.set_title('POS@k vs Removal Steps')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'agnostic_trends_POS.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # NEG subset
    neg_subset = ['NEG@1', 'NEG@5', 'NEG@10']
    neg_data = series(neg_subset)
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, v in neg_data.items():
        ax.plot(steps, v, marker='o', linewidth=2, label=k)
    ax.set_xlabel('Removal Steps (bins)')
    ax.set_ylabel('Score')
    ax.set_title('NEG@k vs Removal Steps')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'agnostic_trends_NEG.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # DEL / INS / NDCG
    other_subset = ['DEL', 'INS', 'NDCG']
    other_data = series(other_subset)
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, v in other_data.items():
        ax.plot(steps, v, marker='o', linewidth=2, label=k)
    ax.set_xlabel('Removal Steps (bins)')
    ax.set_ylabel('Score')
    ax.set_title('DEL / INS / NDCG vs Removal Steps')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'agnostic_trends_OTHER.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def heatmap_step(agg: Dict[int, Dict[str, float]], step: int, output_dir: Path):
    metrics_order = [
        'DEL', 'INS', 'NDCG',
        'POS@1', 'POS@5', 'POS@10', 'POS@20',
        'NEG@1', 'NEG@5', 'NEG@10', 'NEG@20',
    ]
    row = [agg.get(step, {}).get(m, np.nan) for m in metrics_order]
    df = pd.DataFrame([row], index=['NCF+DiceRec'], columns=metrics_order)
    plt.figure(figsize=(12, 2.5))
    sns.heatmap(df, annot=True, cmap='RdYlBu_r', center=0.5, fmt='.3f',
                cbar_kws={'label': 'Metric'})
    plt.title(f'Metrics at Step {step}')
    plt.tight_layout()
    plt.savefig(output_dir / f'agnostic_heatmap_step_{step}.png', dpi=300, bbox_inches='tight')
    plt.close()


def summary_csv_png(agg: Dict[int, Dict[str, float]], step: int, output_dir: Path):
    data = agg.get(step, {})
    if not data:
        return
    df = pd.DataFrame([data], index=['NCF+DiceRec'])
    df.to_csv(output_dir / f'agnostic_summary_step_{step}.csv')

    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.axis('off')
    table = ax.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.3)
    plt.title(f'Summary at Step {step}')
    fig.tight_layout()
    fig.savefig(output_dir / f'agnostic_summary_step_{step}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Visualize model-agnostic eval results (single recommender).')
    parser.add_argument('--results_pkl', required=True, type=str,
                        help='Path to <recommender>_eval_results.pkl produced by eval_embedding_diffusion_agnostic.py')
    parser.add_argument('--output_dir', required=True, type=str,
                        help='Directory to save figures')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_user = load_eval_results(args.results_pkl)
    agg = aggregate_results(per_user)

    plot_line_groups(agg, out_dir)
    # Use step 10 as a standard comparison point if available
    target_step = 10 if 10 in agg else max(agg.keys())
    heatmap_step(agg, target_step, out_dir)
    summary_csv_png(agg, target_step, out_dir)

    print(f"Saved visualizations to: {out_dir}")


if __name__ == '__main__':
    main()
