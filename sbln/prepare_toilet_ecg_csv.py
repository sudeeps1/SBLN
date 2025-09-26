#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare ML-ready CSV from the tOLIet experimental ECG dataset.

For each file in ECG_EXP, this script:
- Reads A1–A4 raw channels and converts them to mV
- Computes robust time, frequency, rhythm, bandpower, and Hjorth features per sensor
- Optionally computes features on sliding windows (more rows, captures dynamics)
- Joins with subject metadata from DataSet.csv (Age, Weight, Height, Gender, Observations)
- Writes a tabular CSV with one row per recording (per file) or per window

Usage (from repository root):
    python -m sbln.prepare_toilet_ecg_csv \
      --dataset-root "data/toliet-single-lead-thigh-based-electrocardiography-using-polimeric-dry-electrodes-1.0.0/toliet-single-lead-thigh-based-electrocardiography-using-polimeric-dry-electrodes-1.0.0" \
      --output "data/toliet_single_lead_ecg_features.csv"

With windowing (example: 5s windows, 2s stride):
    python -m sbln.prepare_toilet_ecg_csv --window-seconds 5 --window-stride-seconds 2 \
      --output "data/toilet_ecg_features_windowed.csv"

Notes:
- Sampling rate assumed to be 1000 Hz (as per file header)
- biosppy is used to detect R-peaks for rhythm features
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def convert_raw_to_millivolts(raw_values: List[int]) -> np.ndarray:
    """Convert raw ADC values to millivolts, matching dataset provided script formula.

    Conversion from `Script/read_ecg_data.py`:
      sig = (1024 - raw)/1024 - 0.5
      sig = sig * (33/11)
    """
    int_values = list(map(int, raw_values))
    signal = 1024 - np.asarray(int_values, dtype=float)
    signal = (signal / 1024.0 - 0.5) * (33.0 / 11.0)
    return signal


def compute_basic_stats(signal: np.ndarray) -> Dict[str, float]:
    if signal.size == 0:
        return {k: np.nan for k in [
            "mean", "std", "min", "max", "median", "iqr", "rms",
            "p25", "p75", "skew", "kurtosis"]}

    x = signal.astype(float)
    mean = float(np.mean(x))
    std = float(np.std(x))
    if np.isnan(std) or std == 0.0:
        std = 1e-12
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    centered = x - mean
    rms = float(np.sqrt(np.mean(x * x)))
    skew = float(np.mean((centered / std) ** 3))
    kurt = float(np.mean((centered / std) ** 4))  # not excess
    return {
        "mean": float(mean),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(q50),
        "iqr": float(q75 - q25),
        "rms": rms,
        "p25": float(q25),
        "p75": float(q75),
        "skew": skew,
        "kurtosis": kurt,
    }


def compute_frequency_features(signal: np.ndarray, fs: float) -> Dict[str, float]:
    if signal.size == 0:
        return {k: np.nan for k in [
            "spec_centroid_hz", "dominant_freq_hz", "spectral_entropy"]}

    x = signal.astype(float)
    # Remove DC component and apply simple Hann window
    x = x - np.median(x)
    window = np.hanning(x.size)
    xw = x * window

    # Real FFT
    spectrum = np.fft.rfft(xw)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(xw.size, d=1.0 / fs)

    # Avoid the zero-frequency bin for centroid/entropy
    if power.size > 1:
        power[0] = 0.0

    total_power = float(np.sum(power))
    if total_power <= 0.0:
        return {k: np.nan for k in [
            "spec_centroid_hz", "dominant_freq_hz", "spectral_entropy"]}

    # Spectral centroid
    centroid = float(np.sum(freqs * power) / total_power)

    # Dominant frequency
    dom_idx = int(np.argmax(power))
    dominant_freq = float(freqs[dom_idx])

    # Spectral entropy (normalized)
    p = power / total_power
    # Ensure numerical stability
    p = np.clip(p, 1e-16, 1.0)
    entropy = float(-np.sum(p * np.log(p)) / np.log(p.size))

    return {
        "spec_centroid_hz": centroid,
        "dominant_freq_hz": dominant_freq,
        "spectral_entropy": entropy,
    }


def _welch_psd(signal: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.signal import welch
    x = signal.astype(float)
    if x.size < 16:
        # Too short; return empty
        return np.array([]), np.array([])
    nperseg = int(min(4096, max(256, 2 ** int(np.floor(np.log2(x.size)) - 1))))
    freqs, psd = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, scaling="density")
    return freqs, psd


def compute_bandpower_features(signal: np.ndarray, fs: float) -> Dict[str, float]:
    """Compute absolute and relative bandpowers over ECG-relevant bands.

    Bands (Hz): 0.5–4, 4–15, 15–40. Values outside 0.5–40 are often noise in ECG.
    """
    freqs, psd = _welch_psd(signal - np.median(signal), fs)
    if freqs.size == 0:
        return {k: np.nan for k in [
            "bp_0_4", "bp_4_15", "bp_15_40", "bp_rel_0_4", "bp_rel_4_15", "bp_rel_15_40", "bp_total"]}

    def bandpower(f: np.ndarray, p: np.ndarray, low: float, high: float) -> float:
        idx = (f >= low) & (f < high)
        if not np.any(idx):
            return 0.0
        return float(np.trapz(p[idx], f[idx]))

    total = bandpower(freqs, psd, 0.5, 40.0)
    bp_0_4 = bandpower(freqs, psd, 0.5, 4.0)
    bp_4_15 = bandpower(freqs, psd, 4.0, 15.0)
    bp_15_40 = bandpower(freqs, psd, 15.0, 40.0)
    if total <= 0:
        rel_0_4 = rel_4_15 = rel_15_40 = np.nan
    else:
        rel_0_4 = float(bp_0_4 / total)
        rel_4_15 = float(bp_4_15 / total)
        rel_15_40 = float(bp_15_40 / total)

    return {
        "bp_total": float(total),
        "bp_0_4": float(bp_0_4),
        "bp_4_15": float(bp_4_15),
        "bp_15_40": float(bp_15_40),
        "bp_rel_0_4": rel_0_4,
        "bp_rel_4_15": rel_4_15,
        "bp_rel_15_40": rel_15_40,
    }


def compute_hjorth(signal: np.ndarray) -> Dict[str, float]:
    """Compute Hjorth mobility and complexity."""
    x = signal.astype(float)
    if x.size < 3:
        return {"hjorth_mobility": np.nan, "hjorth_complexity": np.nan}
    var0 = float(np.var(x))
    if var0 <= 0:
        return {"hjorth_mobility": 0.0, "hjorth_complexity": np.nan}
    dx = np.diff(x)
    var1 = float(np.var(dx))
    mobility = float(np.sqrt(var1 / var0))
    ddx = np.diff(dx)
    var2 = float(np.var(ddx))
    mobility_dx = float(np.sqrt(var2 / (var1 if var1 > 0 else 1e-12)))
    complexity = float(mobility_dx / (mobility if mobility > 0 else 1e-12))
    return {"hjorth_mobility": mobility, "hjorth_complexity": complexity}


def compute_rhythm_features(signal: np.ndarray, fs: float) -> Dict[str, float]:
    """Compute R-peak based features using biosppy if available.

    Returns NaNs if biosppy is unavailable or detection fails.
    """
    try:
        from biosppy.signals import ecg as ecg_module  # lazy import
    except Exception:
        return {k: np.nan for k in [
            "num_rpeaks", "rr_mean_s", "rr_std_s", "hr_mean_bpm", "hr_std_bpm"]}

    try:
        out = ecg_module.ecg(signal=signal.astype(float), sampling_rate=fs, show=False)
        rpeaks = out.get("rpeaks", None)
        if rpeaks is None or len(rpeaks) < 2:
            return {k: np.nan for k in [
                "num_rpeaks", "rr_mean_s", "rr_std_s", "hr_mean_bpm", "hr_std_bpm"]}

        rpeaks = np.asarray(rpeaks, dtype=float)
        rr = np.diff(rpeaks) / fs
        if rr.size == 0:
            return {k: np.nan for k in [
                "num_rpeaks", "rr_mean_s", "rr_std_s", "hr_mean_bpm", "hr_std_bpm"]}

        rr_mean = float(np.mean(rr))
        rr_std = float(np.std(rr))
        hr = 60.0 / rr
        hr_mean = float(np.mean(hr))
        hr_std = float(np.std(hr))
        return {
            "num_rpeaks": float(rpeaks.size),
            "rr_mean_s": rr_mean,
            "rr_std_s": rr_std,
            "hr_mean_bpm": hr_mean,
            "hr_std_bpm": hr_std,
        }
    except Exception:
        return {k: np.nan for k in [
            "num_rpeaks", "rr_mean_s", "rr_std_s", "hr_mean_bpm", "hr_std_bpm"]}


def extract_features_for_signal(signal_mv: np.ndarray, fs: float) -> Dict[str, float]:
    features: Dict[str, float] = {}
    features.update(compute_basic_stats(signal_mv))
    features.update(compute_frequency_features(signal_mv, fs))
    features.update(compute_bandpower_features(signal_mv, fs))
    features.update(compute_hjorth(signal_mv))
    features.update(compute_rhythm_features(signal_mv, fs))
    # Signal length features
    features.update({
        "num_samples": float(signal_mv.size),
        "duration_s": float(signal_mv.size) / fs if signal_mv.size else np.nan,
    })
    return features


def read_experimental_file(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read ECG_EXP text file and return A1–A4 in mV.

    The files have a header beginning with '#'; numpy.loadtxt will skip it by default.
    Columns of interest (0-indexed): 5=A1, 6=A2, 7=A3, 8=A4
    """
    data = np.loadtxt(path)
    # Defensive: ensure we have at least 9 columns
    if data.ndim == 1:
        data = data.reshape(-1, 11)
    a1_raw = [int(x) for x in data[:, 5]]
    a2_raw = [int(x) for x in data[:, 6]]
    a3_raw = [int(x) for x in data[:, 7]]
    a4_raw = [int(x) for x in data[:, 8]]

    a1 = convert_raw_to_millivolts(a1_raw)
    a2 = convert_raw_to_millivolts(a2_raw)
    a3 = convert_raw_to_millivolts(a3_raw)
    a4 = convert_raw_to_millivolts(a4_raw)
    return a1, a2, a3, a4


def sanitize_observation(obs: str) -> str:
    if not isinstance(obs, str):
        return ""
    return obs.strip()


def parse_observation_labels(obs: str) -> Dict[str, str | int]:
    """Extract labels from observations text.

    - Primary label: multi-class string among {"AF", "LBBB", "Cardiomyopathy", "AF+LBBB", ""}
    - Binary flags: label_af_binary, label_lbbb_binary, label_cardiomyopathy_binary
    - label_any_observation: 1 if any non-empty observation
    """
    text = (obs or "").strip().lower()
    has_any = 1 if len(text) > 0 else 0
    has_af = 1 if ("af" in text or "atrial fibrillation" in text) else 0
    has_lbbb = 1 if ("left bundle branch block" in text or "lbbb" in text) else 0
    has_cardio = 1 if ("cardiomyopathy" in text) else 0

    primary = ""
    if has_af and has_lbbb:
        primary = "AF+LBBB"
    elif has_af:
        primary = "AF"
    elif has_lbbb:
        primary = "LBBB"
    elif has_cardio:
        primary = "Cardiomyopathy"

    return {
        "label": primary,
        "label_af_binary": has_af,
        "label_lbbb_binary": has_lbbb,
        "label_cardiomyopathy_binary": has_cardio,
        "label_any_observation": has_any,
    }


def prepare_csv(dataset_root: str, output_csv: str, fs: float = 1000.0) -> None:
    """Prepare CSV with either per-record or per-window features."""
    exp_dir = os.path.join(dataset_root, "ECG_EXP")
    meta_csv = os.path.join(dataset_root, "DataSet.csv")

    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"ECG_EXP directory not found: {exp_dir}")
    if not os.path.isfile(meta_csv):
        raise FileNotFoundError(f"Metadata file not found: {meta_csv}")

    # Read metadata (semicolon-separated, with extra empty columns). Keep only first 6 columns.
    meta_df_raw = pd.read_csv(meta_csv, sep=";", dtype=str, engine="python")
    # Normalize column names to consistent identifiers
    meta_df = meta_df_raw.iloc[:, :6].copy()
    meta_df.columns = [
        "ID",
        "Age",
        "Weight",
        "Height",
        "Gender",
        "Observations",
    ]
    meta_df["ID"] = meta_df["ID"].astype(str)
    meta_df["Observations"] = meta_df["Observations"].apply(sanitize_observation)

    rows: List[Dict[str, float | str]] = []
    file_list = [f for f in os.listdir(exp_dir) if f.lower().endswith(".txt")]
    file_list.sort(key=lambda x: (len(x), x))

    for fname in file_list:
        file_id = os.path.splitext(fname)[0]
        path = os.path.join(exp_dir, fname)

        # Join metadata row (may be missing for some file ids)
        meta_row = meta_df.loc[meta_df["ID"] == file_id]
        if meta_row.empty:
            # Try to handle ids like '01' vs '1'
            meta_row = meta_df.loc[meta_df["ID"].str.lstrip("0") == file_id.lstrip("0")]

        try:
            a1, a2, a3, a4 = read_experimental_file(path)
        except Exception as e:
            # Record an error entry and continue
            rows.append({
                "record_id": file_id,
                "source_file": fname,
                "fs_hz": fs,
                "error": f"read_failed: {e}",
            })
            continue

        def meta_dict() -> Dict[str, str | int]:
            if not meta_row.empty:
                obs_val = str(meta_row.iloc[0]["Observations"]) if "Observations" in meta_row.columns else ""
                labels = parse_observation_labels(obs_val)
                return {
                    "SubjectID": meta_row.iloc[0]["ID"],
                    "Age": meta_row.iloc[0]["Age"],
                    "Weight": meta_row.iloc[0]["Weight"],
                    "Height": meta_row.iloc[0]["Height"],
                    "Gender": meta_row.iloc[0]["Gender"],
                    "Observations": obs_val,
                    **labels,
                }
            return {
                "SubjectID": "",
                "Age": "", "Weight": "", "Height": "", "Gender": "", "Observations": "",
                "label": "", "label_af_binary": 0, "label_lbbb_binary": 0, "label_cardiomyopathy_binary": 0,
                "label_any_observation": 0,
            }

        # Decide between per-record aggregate and windowed
        if WINDOW_SECONDS <= 0:
            feat_a1 = {f"a1_{k}": v for k, v in extract_features_for_signal(a1, fs).items()}
            feat_a2 = {f"a2_{k}": v for k, v in extract_features_for_signal(a2, fs).items()}
            feat_a3 = {f"a3_{k}": v for k, v in extract_features_for_signal(a3, fs).items()}
            feat_a4 = {f"a4_{k}": v for k, v in extract_features_for_signal(a4, fs).items()}

            sensors_mean_amp = float(np.mean([
                np.mean(a1) if a1.size else np.nan,
                np.mean(a2) if a2.size else np.nan,
                np.mean(a3) if a3.size else np.nan,
                np.mean(a4) if a4.size else np.nan,
            ]))
            sensors_std_amp = float(np.mean([
                np.std(a1) if a1.size else np.nan,
                np.std(a2) if a2.size else np.nan,
                np.std(a3) if a3.size else np.nan,
                np.std(a4) if a4.size else np.nan,
            ]))

            row: Dict[str, float | str] = {
                "record_id": file_id,
                "source_file": fname,
                "fs_hz": fs,
                "mean_amp_mv_all": sensors_mean_amp,
                "std_amp_mv_all": sensors_std_amp,
            }
            row.update(meta_dict())
            row.update(feat_a1)
            row.update(feat_a2)
            row.update(feat_a3)
            row.update(feat_a4)
            rows.append(row)
        else:
            # Windowed processing
            win = int(round(WINDOW_SECONDS * fs))
            stride = int(round(WINDOW_STRIDE_SECONDS * fs))
            if stride <= 0:
                stride = win

            def windows(x: np.ndarray) -> Iterable[Tuple[int, int, np.ndarray]]:
                n = x.size
                if n < win:
                    yield 0, n, x
                    return
                start = 0
                widx = 0
                while start + win <= n:
                    segment = x[start:start + win]
                    yield widx, start, segment
                    widx += 1
                    start += stride

            # We iterate over window indices from A1 and slice all sensors accordingly for alignment
            total_len = int(min(a1.size, a2.size, a3.size, a4.size))
            a1 = a1[:total_len]
            a2 = a2[:total_len]
            a3 = a3[:total_len]
            a4 = a4[:total_len]
            widx = 0
            start = 0
            n = total_len
            if n < win:
                # Single window covering entire signal
                feat_a1 = {f"a1_{k}": v for k, v in extract_features_for_signal(a1, fs).items()}
                feat_a2 = {f"a2_{k}": v for k, v in extract_features_for_signal(a2, fs).items()}
                feat_a3 = {f"a3_{k}": v for k, v in extract_features_for_signal(a3, fs).items()}
                feat_a4 = {f"a4_{k}": v for k, v in extract_features_for_signal(a4, fs).items()}
                row = {
                    "record_id": file_id,
                    "source_file": fname,
                    "fs_hz": fs,
                    "window_index": 0,
                    "window_start_s": 0.0,
                    "window_end_s": float(n) / fs,
                }
                row.update(meta_dict())
                row.update(feat_a1)
                row.update(feat_a2)
                row.update(feat_a3)
                row.update(feat_a4)
                rows.append(row)
            else:
                while start + win <= n:
                    seg_a1 = a1[start:start + win]
                    seg_a2 = a2[start:start + win]
                    seg_a3 = a3[start:start + win]
                    seg_a4 = a4[start:start + win]

                    feat_a1 = {f"a1_{k}": v for k, v in extract_features_for_signal(seg_a1, fs).items()}
                    feat_a2 = {f"a2_{k}": v for k, v in extract_features_for_signal(seg_a2, fs).items()}
                    feat_a3 = {f"a3_{k}": v for k, v in extract_features_for_signal(seg_a3, fs).items()}
                    feat_a4 = {f"a4_{k}": v for k, v in extract_features_for_signal(seg_a4, fs).items()}

                    row = {
                        "record_id": file_id,
                        "source_file": fname,
                        "fs_hz": fs,
                        "window_index": widx,
                        "window_start_s": float(start) / fs,
                        "window_end_s": float(start + win) / fs,
                    }
                    row.update(meta_dict())
                    row.update(feat_a1)
                    row.update(feat_a2)
                    row.update(feat_a3)
                    row.update(feat_a4)
                    rows.append(row)

                    widx += 1
                    start += stride

    df = pd.DataFrame(rows)
    # Sort columns: identifiers, metadata, label, then features
    id_cols = ["record_id", "source_file", "fs_hz"]
    if WINDOW_SECONDS > 0:
        id_cols += ["window_index", "window_start_s", "window_end_s"]
    meta_cols = ["SubjectID", "Age", "Weight", "Height", "Gender", "Observations"]
    label_cols = ["label", "label_af_binary", "label_lbbb_binary", "label_cardiomyopathy_binary", "label_any_observation"]
    feature_cols = sorted([c for c in df.columns if c not in set(id_cols + meta_cols + label_cols + ["error"])])
    ordered_cols = id_cols + meta_cols + label_cols + feature_cols

    # Place error column at the end if present
    if "error" in df.columns:
        ordered_cols = ordered_cols + ["error"]

    df = df[ordered_cols]
    # Write CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {len(df)} rows to {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ML-ready CSV from tOLIet ECG dataset")
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=False,
        default=(
            "data/toliet-single-lead-thigh-based-electrocardiography-using-polimeric-dry-"
            "electrodes-1.0.0/toliet-single-lead-thigh-based-electrocardiography-using-"
            "polimeric-dry-electrodes-1.0.0"
        ),
        help="Path to the dataset root containing ECG_EXP, ECG_REF, and DataSet.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default="data/toilet_ecg_features.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--fs",
        type=float,
        required=False,
        default=1000.0,
        help="Sampling frequency in Hz",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        required=False,
        default=0.0,
        help="If > 0, compute features on sliding windows of this length (seconds)",
    )
    parser.add_argument(
        "--window-stride-seconds",
        type=float,
        required=False,
        default=0.0,
        help="Stride between windows in seconds (defaults to window length if 0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global WINDOW_SECONDS, WINDOW_STRIDE_SECONDS
    WINDOW_SECONDS = float(args.window_seconds) if args.window_seconds else 0.0
    WINDOW_STRIDE_SECONDS = float(args.window_stride_seconds) if args.window_stride_seconds else 0.0
    prepare_csv(dataset_root=args.dataset_root, output_csv=args.output, fs=args.fs)


if __name__ == "__main__":
    main()


