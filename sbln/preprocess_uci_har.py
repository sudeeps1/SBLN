import os
import pandas as pd
from typing import Optional, Tuple


def _find_har_base_dir(root_dir: str = "data") -> Optional[str]:
    """
    Try to locate the UCI HAR base directory by searching for 'features.txt'.
    Common layouts:
      data/UCI HAR Dataset/
        - features.txt
        - activity_labels.txt
        - train/X_train.txt, y_train.txt, subject_train.txt
        - test/X_test.txt, y_test.txt, subject_test.txt
    Returns the path to the directory containing features.txt or None if not found.
    """
    candidates = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'features.txt' in filenames and 'activity_labels.txt' in filenames:
            candidates.append(dirpath)
    if not candidates:
        return None
    # Prefer the shortest path that matches (more likely to be the root)
    return sorted(candidates, key=len)[0]


def _load_har_split(base_dir: str, split: str, feature_names: list[str]) -> pd.DataFrame:
    """
    Load a split (train/test) and return a DataFrame with columns:
    [feature columns..., 'subject', 'Activity', 'split']
    """
    x_path = os.path.join(base_dir, split, f"X_{split}.txt")
    y_path = os.path.join(base_dir, split, f"y_{split}.txt")
    s_path = os.path.join(base_dir, split, f"subject_{split}.txt")

    X = pd.read_csv(x_path, delim_whitespace=True, header=None)
    X.columns = feature_names
    y = pd.read_csv(y_path, delim_whitespace=True, header=None)[0]
    subj = pd.read_csv(s_path, delim_whitespace=True, header=None)[0]

    df = X.copy()
    df['subject'] = subj.values
    df['Activity'] = y.values
    df['split'] = split
    return df


def preprocess_uci_har(root_dir: str = "data", output_csv: str = "data/uci_har.csv") -> Tuple[str, pd.DataFrame]:
    """
    Preprocess the UCI HAR dataset into a single CSV usable by the SBLN trainer.

    - Searches for the base directory containing features.txt under root_dir
    - Loads train and test splits, merges them, and maps activity ids to labels
    - Writes a CSV with columns: 561 features, 'subject', 'Activity' (string label), 'split'

    Returns: (output_csv_path, dataframe)
    """
    base_dir = _find_har_base_dir(root_dir)
    if base_dir is None:
        raise FileNotFoundError(
            f"Could not locate UCI HAR dataset under '{root_dir}'. Expected 'features.txt' and 'activity_labels.txt'."
        )

    # Load feature names and activity labels
    features_path = os.path.join(base_dir, 'features.txt')
    activity_labels_path = os.path.join(base_dir, 'activity_labels.txt')

    features_df = pd.read_csv(features_path, delim_whitespace=True, header=None, names=['idx', 'name'])
    # Sanitize feature names to be CSV/column friendly
    feature_names = (
        features_df['name']
        .str.replace('[()\-/,]', '_', regex=True)
        .str.replace('__+', '_', regex=True)
        .str.strip('_')
        .tolist()
    )

    activity_map_df = pd.read_csv(activity_labels_path, delim_whitespace=True, header=None, names=['id', 'label'])
    activity_map = dict(zip(activity_map_df['id'].astype(int), activity_map_df['label'].astype(str)))

    # Load splits
    train_df = _load_har_split(base_dir, 'train', feature_names)
    test_df = _load_har_split(base_dir, 'test', feature_names)
    all_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # Map numeric activity id to label string for readability
    all_df['Activity'] = all_df['Activity'].astype(int).map(activity_map)

    # Save CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    all_df.to_csv(output_csv, index=False)
    return output_csv, all_df


if __name__ == "__main__":
    out_path, df = preprocess_uci_har()
    print(f"Saved preprocessed UCI HAR to {out_path} with shape {df.shape}")

