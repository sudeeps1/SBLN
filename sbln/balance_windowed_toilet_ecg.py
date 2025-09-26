#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Balance windowed toilet ECG features into a 50/50 labeled/unlabeled dataset.

This script:
- Reads a windowed features CSV produced by `sbln.prepare_toilet_ecg_csv` (with --window-seconds > 0)
- Selects ALL rows where `label_any_observation == 1` as the labeled half
- Randomly samples the same number of rows from the unlabeled pool
- Concatenates and shuffles to produce a balanced CSV

Usage:
    python -m sbln.balance_windowed_toilet_ecg \
      --input data/toilet_ecg_features_windowed.csv \
      --output data/toilet_ecg_features_windowed_balanced.csv \
      --seed 42
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd


def balance_csv(input_csv: str, output_csv: str, seed: int = 42) -> None:
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if "label_any_observation" not in df.columns:
        raise ValueError("Input CSV is missing 'label_any_observation'. Regenerate with newer prepare script.")

    labeled = df[df["label_any_observation"] == 1]
    unlabeled = df[df["label_any_observation"] != 1]

    n_labeled = len(labeled)
    if n_labeled == 0:
        raise ValueError("No labeled windows found. Ensure your input is windowed and labels were parsed.")

    if len(unlabeled) == 0:
        raise ValueError("No unlabeled windows found to balance against.")

    n_sample = min(n_labeled, len(unlabeled))
    rng = np.random.default_rng(seed)
    unlabeled_sample = unlabeled.sample(n=n_sample, random_state=seed)

    balanced = pd.concat([labeled, unlabeled_sample], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    balanced.to_csv(output_csv, index=False)
    print(f"Balanced CSV written to {output_csv} with {len(balanced)} rows (labeled={n_labeled}, unlabeled_sampled={n_sample}).")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Balance windowed ECG features into 50/50 labeled/unlabeled")
    ap.add_argument("--input", type=str, required=True, help="Path to windowed features CSV")
    ap.add_argument("--output", type=str, required=True, help="Path for balanced CSV")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    balance_csv(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()



