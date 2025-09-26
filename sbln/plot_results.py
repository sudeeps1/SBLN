import argparse
import os
import re
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())


def load_results(csv_path: str) -> pd.DataFrame:
    # Robust CSV loading with fallbacks
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        try:
            # Try python engine with on_bad_lines skipped
            df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')
        except Exception:
            # Try automatic sep detection
            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                sample = ''.join([next(f) for _ in range(50)])
            sep = ','
            if '\t' in sample and sample.count('\t') > sample.count(','):
                sep = '\t'
            df = pd.read_csv(csv_path, sep=sep, engine='python', on_bad_lines='skip')

    if 'Model' not in df.columns:
        raise ValueError("Results CSV must contain a 'Model' column")
    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]
    # Ensure 'Model' is string and stripped
    df['Model'] = df['Model'].astype(str).str.strip()
    # Drop fully-empty dataset columns and rows without a model name
    df = df.loc[df['Model'].str.len() > 0]
    empty_cols = [c for c in df.columns if c != 'Model' and df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
    return df


def to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col == 'Model':
            continue
        out[col] = pd.to_numeric(out[col], errors='coerce')
    return out


def plot_heatmap_all(df: pd.DataFrame, out_path: str):
    pivot = df.set_index('Model')
    pivot = to_numeric(pivot)
    plt.figure(figsize=(max(8, 1.2 * len(pivot.columns)), max(4, 0.6 * len(pivot.index))))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis', cbar_kws={'label': 'Score'})
    plt.title('Model Performance by Dataset')
    plt.ylabel('Model')
    plt.xlabel('Dataset')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_bar_for_dataset(df: pd.DataFrame, dataset_col: str, out_path: str):
    if dataset_col not in df.columns:
        raise ValueError(f"Dataset column '{dataset_col}' not found in CSV")
    sub = df[['Model', dataset_col]].copy()
    sub[dataset_col] = pd.to_numeric(sub[dataset_col], errors='coerce')
    sub = sub.dropna(subset=[dataset_col])
    sub = sub.sort_values(dataset_col, ascending=False)

    plt.figure(figsize=(8, 4.5))
    sns.barplot(x=dataset_col, y='Model', data=sub, palette='Blues_r')
    for i, v in enumerate(sub[dataset_col].values):
        plt.text(v, i, f" {v:.3f}", va='center')
    plt.title(f'Model Comparison on {dataset_col}')
    plt.xlabel('Score')
    plt.ylabel('Model')
    plt.xlim(left=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_average_rank(df: pd.DataFrame, out_path: str):
    numeric = to_numeric(df.set_index('Model'))
    # Compute ranks per dataset column (higher is better)
    ranks = numeric.rank(ascending=False, axis=0, method='average')
    avg_rank = ranks.mean(axis=1).sort_values()
    plt.figure(figsize=(8, 4.5))
    sns.barplot(x=avg_rank.values, y=avg_rank.index, palette='mako')
    for i, v in enumerate(avg_rank.values):
        plt.text(v, i, f" {v:.2f}", va='center')
    plt.title('Average Rank Across Datasets (lower is better)')
    plt.xlabel('Average Rank')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot figures from results_model.csv')
    parser.add_argument('--csv', type=str, default='results_model.csv', help='Path to results CSV')
    parser.add_argument('--dataset', type=str, nargs='*', help='Specific dataset column(s) to plot')
    parser.add_argument('--outdir', type=str, default='results_figures', help='Output directory for figures')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_results(args.csv)

    # Heatmap across all datasets
    heatmap_path = os.path.join(args.outdir, 'heatmap_all.png')
    plot_heatmap_all(df, heatmap_path)

    # Average rank across datasets
    rank_path = os.path.join(args.outdir, 'average_rank.png')
    plot_average_rank(df, rank_path)

    # Per-dataset bar plots (selected or all)
    dataset_cols: List[str]
    if args.dataset and len(args.dataset) > 0:
        dataset_cols = args.dataset
    else:
        dataset_cols = [c for c in df.columns if c != 'Model']

    for col in dataset_cols:
        out_path = os.path.join(args.outdir, f"bar_{sanitize_filename(col)}.png")
        try:
            plot_bar_for_dataset(df, col, out_path)
        except Exception as e:
            # Skip invalid columns
            print(f"Skipping column '{col}': {e}")

    print(f"Saved figures to {os.path.abspath(args.outdir)}")


if __name__ == '__main__':
    main()


