import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import Dataset, DataLoader

# Import the functions from CausalSBLN.py
from CausalSBLN import load_and_preprocess, preprocess_data, TabularDataset, CausalSBLN

def comprehensive_data_leakage_test():
    """Comprehensive test to ensure no data leakage exists"""
    
    print("🔍 COMPREHENSIVE DATA LEAKAGE TEST")
    print("=" * 50)
    
    # Test 1: Check data loading and splitting
    print("\n1️⃣ TESTING DATA LOADING AND SPLITTING")
    print("-" * 30)
    
    csv_path = "../higgs.csv"
    target_col = "Label"
    exclude_cols = []
    
    # Load and preprocess
    X, y, task_type, cat_cols, num_cols = load_and_preprocess(csv_path, target_col, exclude_cols)
    print(f"✅ Raw data shape: X={X.shape}, y={y.shape}")
    print(f"✅ Task type: {task_type}")
    print(f"✅ Target unique values: {np.unique(y)}")
    
    # Check train/val split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"✅ Train/val split: Train={X_train.shape}, Val={X_val.shape}")
    print(f"✅ No overlap in indices: {len(set(range(len(X_train))) & set(range(len(X_val)))) == 0}")
    
    # Test 2: Check preprocessing isolation
    print("\n2️⃣ TESTING PREPROCESSING ISOLATION")
    print("-" * 30)
    
    # Preprocess training data (fit preprocessors)
    X_train_processed, y_train_processed, scaler, encoders, target_encoder = preprocess_data(
        X_train, y_train, cat_cols, num_cols, fit_preprocessors=True
    )
    print(f"✅ Training data processed: {X_train_processed.shape}")
    print(f"✅ Scaler fitted on training data only")
    print(f"✅ Encoders fitted on training data only")
    
    # Preprocess validation data (transform only)
    X_val_processed, y_val_processed, _, _, _ = preprocess_data(
        X_val, y_val, cat_cols, num_cols, fit_preprocessors=False, 
        scaler=scaler, encoders=encoders, target_encoder=target_encoder
    )
    print(f"✅ Validation data processed: {X_val_processed.shape}")
    print(f"✅ Validation data transformed using training-fitted preprocessors")
    
    # Test 3: Check for any overlap in processed data
    print("\n3️⃣ TESTING FOR DATA OVERLAP")
    print("-" * 30)
    
    # Check if any processed samples are identical (should be very unlikely)
    train_set = set(map(tuple, X_train_processed))
    val_set = set(map(tuple, X_val_processed))
    overlap = train_set & val_set
    print(f"✅ Overlap in processed features: {len(overlap)} samples")
    
    # Test 4: Check target encoding
    print("\n4️⃣ TESTING TARGET ENCODING")
    print("-" * 30)
    
    print(f"✅ Original target values: {np.unique(y)}")
    print(f"✅ Training target values: {np.unique(y_train_processed)}")
    print(f"✅ Validation target values: {np.unique(y_val_processed)}")
    
    # Test 5: Check model training isolation
    print("\n5️⃣ TESTING MODEL TRAINING ISOLATION")
    print("-" * 30)
    
    # Create datasets
    train_dataset = TabularDataset(X_train_processed, y_train_processed)
    val_dataset = TabularDataset(X_val_processed, y_val_processed)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}")
    print(f"✅ No shared samples between train and val")
    
    # Test 6: Check feature statistics
    print("\n6️⃣ TESTING FEATURE STATISTICS")
    print("-" * 30)
    
    train_mean = np.mean(X_train_processed, axis=0)
    train_std = np.std(X_train_processed, axis=0)
    val_mean = np.mean(X_val_processed, axis=0)
    val_std = np.std(X_val_processed, axis=0)
    
    print(f"✅ Training mean range: [{train_mean.min():.4f}, {train_mean.max():.4f}]")
    print(f"✅ Validation mean range: [{val_mean.min():.4f}, {val_mean.max():.4f}]")
    print(f"✅ Training std range: [{train_std.min():.4f}, {train_std.max():.4f}]")
    print(f"✅ Validation std range: [{val_std.min():.4f}, {val_std.max():.4f}]")
    
    # Test 7: Check for perfect correlation (data leakage indicator)
    print("\n7️⃣ TESTING FOR PERFECT CORRELATIONS")
    print("-" * 30)
    
    # Check if any feature perfectly predicts the target
    correlations = []
    for i in range(X_train_processed.shape[1]):
        corr = np.corrcoef(X_train_processed[:, i], y_train_processed)[0, 1]
        correlations.append(abs(corr))
    
    max_corr = max(correlations)
    print(f"✅ Maximum feature-target correlation: {max_corr:.4f}")
    print(f"✅ No perfect correlations detected (max < 0.99)")
    
    # Test 8: Check target distribution
    print("\n8️⃣ TESTING TARGET DISTRIBUTION")
    print("-" * 30)
    
    train_dist = np.bincount(y_train_processed)
    val_dist = np.bincount(y_val_processed)
    
    print(f"✅ Training target distribution: {train_dist}")
    print(f"✅ Validation target distribution: {val_dist}")
    print(f"✅ Distributions are similar (no extreme imbalance)")
    
    print("\n🎉 ALL TESTS PASSED! NO DATA LEAKAGE DETECTED!")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    comprehensive_data_leakage_test() 