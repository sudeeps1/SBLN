import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
from sklearn.feature_extraction.text import HashingVectorizer


# Model Libraries
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from catboost import CatBoostClassifier, CatBoostRegressor

# Deep Learning Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# TabNet
try:
    from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    print("TabNet not available. Install with: pip install pytorch-tabnet")

# FTTransformer
try:
    from pytorch_tabular import TabularModel
    from pytorch_tabular.models import FTTransformer
    from pytorch_tabular.config import ModelConfig, OptimizerConfig, TrainerConfig
    FTTRANSFORMER_AVAILABLE = True
except ImportError:
    FTTRANSFORMER_AVAILABLE = False
    print("FTTransformer not available. Install with: pip install pytorch-tabular")

# AutoGluon
try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False
    print("AutoGluon not available. Install with: pip install autogluon.tabular")

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
from typing import Dict, List, Tuple, Optional, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class DataPreprocessor:
    """Handles data preprocessing with proper train/test separation to prevent data leakage"""
    
    def __init__(self):
        self.cat_encoders = {}
        self.num_scaler = None
        self.num_imputer = None
        self.cat_imputer = None
        self.categorical_columns = []
        self.numerical_columns = []
        self.is_fitted = False
        
    def fit_transform(self, X_train, X_test=None):
        """Fit preprocessors on training data and transform both train and test"""
        # Identify column types
        self.categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Initialize preprocessors
        self.num_imputer = SimpleImputer(strategy='mean')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.num_scaler = StandardScaler()
        
        # Process numerical features
        if self.numerical_columns:
            X_train_num = self.num_imputer.fit_transform(X_train[self.numerical_columns])
            X_train_num = self.num_scaler.fit_transform(X_train_num)
            if X_test is not None:
                X_test_num = self.num_imputer.transform(X_test[self.numerical_columns])
                X_test_num = self.num_scaler.transform(X_test_num)
        else:
            X_train_num = np.empty((len(X_train), 0))
            X_test_num = np.empty((len(X_test), 0)) if X_test is not None else None
        
        # Process categorical features
        if self.categorical_columns:
            X_train_cat_raw = self.cat_imputer.fit_transform(X_train[self.categorical_columns])
            X_train_cat = np.zeros_like(X_train_cat_raw, dtype=np.float32)
            
            # Encode categorical features
            for i, col in enumerate(self.categorical_columns):
                le = LabelEncoder()
                X_train_cat[:, i] = le.fit_transform(X_train_cat_raw[:, i])
                self.cat_encoders[col] = le
            
            if X_test is not None:
                X_test_cat_raw = self.cat_imputer.transform(X_test[self.categorical_columns])
                X_test_cat = np.zeros_like(X_test_cat_raw, dtype=np.float32)
                
                for i, col in enumerate(self.categorical_columns):
                    try:
                        X_test_cat[:, i] = self.cat_encoders[col].transform(X_test_cat_raw[:, i])
                    except ValueError as e:
                        # Handle unseen categories by using a default value
                        logger.warning(f"Unseen categories in column {col}, using default encoding: {str(e)}")
                        X_test_cat[:, i] = len(self.cat_encoders[col].classes_)
        else:
            X_train_cat = np.empty((len(X_train), 0))
            X_test_cat = np.empty((len(X_test), 0)) if X_test is not None else None
        
        # Combine features
        X_train_processed = np.hstack([X_train_num, X_train_cat])
        self.is_fitted = True
        
        if X_test is not None:
            X_test_processed = np.hstack([X_test_num, X_test_cat])
            return X_train_processed, X_test_processed
        
        return X_train_processed
    
    def transform(self, X):
        """Transform data using fitted preprocessors"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Process numerical features
        if self.numerical_columns:
            X_num = self.num_imputer.transform(X[self.numerical_columns])
            X_num = self.num_scaler.transform(X_num)
        else:
            X_num = np.empty((len(X), 0))
        
        # Process categorical features
        if self.categorical_columns:
            X_cat_raw = self.cat_imputer.transform(X[self.categorical_columns])
            X_cat = np.zeros_like(X_cat_raw, dtype=np.float32)
            
            for i, col in enumerate(self.categorical_columns):
                try:
                    X_cat[:, i] = self.cat_encoders[col].transform(X_cat_raw[:, i])
                except ValueError as e:
                    logger.warning(f"Unseen categories in column {col}, using default encoding: {str(e)}")
                    X_cat[:, i] = len(self.cat_encoders[col].classes_)
        else:
            X_cat = np.empty((len(X), 0))
        
        # Combine features
        return np.hstack([X_num, X_cat])

class MLPDataset(Dataset):
    """Dataset for MLP training"""
    def __init__(self, X, y, task_type='classification'):
        self.X = torch.FloatTensor(X)
        if task_type == 'classification':
            self.y = torch.LongTensor(y)
        else:
            self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    """Multi-Layer Perceptron for tabular data"""
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class ModelTrainer:
    """Comprehensive model trainer with data leakage prevention"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.preprocessor = None
        
    def load_data(self, file_path, target_column, test_size=0.2, random_state=42, exclude_columns: Optional[List[str]] = None):
        """Load and split data with proper train/test separation.
        Optionally exclude a list of columns from the features.
        """
        logger.info(f"Loading data from {file_path}")
        
        # Load data
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        else:
            # Handle other formats like .data
            data = pd.read_csv(file_path, sep='\s+', header=None)
            
            # Check if first row looks like header
            first_row = data.iloc[0].astype(str)
            if any('class' in val.lower() or 'id' in val.lower() or 'sequence' in val.lower() for val in first_row):
                logger.info("Detected header row, skipping it...")
                data = pd.read_csv(file_path, sep='\s+', header=0)
            
            # Assume last column is target if no target_column specified
            if target_column is None:
                target_column = data.columns[-1]
        
        # Clean column names (strip leading/trailing spaces)
        data.columns = [str(c).strip() for c in data.columns]
        target_column = str(target_column).strip()

        # Normalize exclude list
        exclude_columns = [c.strip() for c in (exclude_columns or []) if c and c.strip()]
        if exclude_columns:
            logger.info(f"Requested to exclude columns: {exclude_columns}")

        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Target column: {target_column}")
        
        # Check if target column exists
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Clean data - drop rows with all NaNs in features or target
        data = data.dropna(subset=[target_column])
        data = data.dropna(axis=1, how='all')
        
        # Separate features and target
        X = data.drop(columns=[target_column])
        
        # Drop explicitly excluded columns from X
        if exclude_columns:
            to_drop = [c for c in exclude_columns if c in X.columns]
            missing = [c for c in exclude_columns if c not in X.columns]
            if to_drop:
                logger.info(f"Dropping excluded columns from features: {to_drop}")
                X = X.drop(columns=to_drop)
            if missing:
                logger.warning(f"Exclude columns not found in data (ignored): {missing}")
        y = data[target_column].values

        # Drop extremely high-cardinality string columns that behave like IDs/sequences
        obj_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        high_card_cols = []
        n_rows = len(X)
        for col in obj_cols:
            nu = X[col].nunique(dropna=True)
            if nu > 1000 or nu > 0.5 * n_rows:
                high_card_cols.append(col)
        if high_card_cols:
            logger.warning(
                f"Dropping high-cardinality categorical columns (likely IDs/sequences): {high_card_cols}"
            )
            X = X.drop(columns=high_card_cols)

        # If no features remain (e.g., splice: only class/id/sequence), build hashed char-3gram features
        if X.shape[1] == 0:
            # Pick a sequence-like column from original data (excluding target)
            candidate_cols = [c for c in data.columns if c != target_column and data[c].dtype == object]
            if candidate_cols:
                # Choose the column with the largest average string length
                avg_len = {c: data[c].astype(str).str.len().mean() for c in candidate_cols}
                seq_col = max(avg_len, key=avg_len.get)
                logger.warning(f"No structured features left; using hashed 3-gram features from '{seq_col}'")
                hv = HashingVectorizer(analyzer='char', ngram_range=(3, 3), n_features=1024, norm=None, alternate_sign=False)
                X_hashed = hv.transform(data[seq_col].astype(str).values)
                # Replace X with hashed features and drop the sequence column
                X = pd.DataFrame(X_hashed.toarray(), index=data.index, columns=[f"seqhash_{i}" for i in range(X_hashed.shape[1])])
            else:
                raise ValueError("No usable feature columns remain after filtering, and no sequence-like columns found to featurize.")
        
        # Detect task type more robustly (matching CausalSBLN logic)
        unique_vals = np.unique(y)
        
        # Check for sequence-like data (very long strings)
        if y.dtype.kind in 'UO':  # Unicode or Object
            avg_length = np.mean([len(str(val)) for val in y[:100]])  # Sample first 100
            if avg_length > 50:  # Very long strings, likely sequences
                logger.warning(f"Detected sequence-like data (avg length: {avg_length:.1f}). This may not be suitable for standard tabular classification.")
                if len(unique_vals) > 1000:
                    logger.error(f"Too many unique sequences ({len(unique_vals)}) for standard classification. Consider using a different target column.")
                    raise ValueError(f"Dataset has {len(unique_vals)} unique sequences, which is too many for standard classification. Please choose a different target column.")
        
        if y.dtype.kind in 'iufc' and len(unique_vals) > 10:
            self.task_type = 'regression'
            self.target_encoder = None
            logger.info("Regression task")
        elif len(unique_vals) <= 20:
            # Check if all values are integers (for numeric data)
            if y.dtype.kind in 'iufc':
                if np.all(np.mod(unique_vals, 1) == 0):
                    self.task_type = 'classification'
                else:
                    self.task_type = 'regression'
            else:
                # For string/categorical data, always treat as classification
                self.task_type = 'classification'
        else:
            # Default fallback
            self.task_type = 'classification'
        
        # Encode target for classification
        if self.task_type == 'classification':
            if y.dtype.kind not in 'iufc' or (len(unique_vals) <= 20 and np.all(np.mod(unique_vals, 1) == 0)):
                le = LabelEncoder()
                y = le.fit_transform(y)
                self.target_encoder = le
                logger.info(f"Classification task with {len(le.classes_)} classes")
            else:
                self.target_encoder = None
                logger.info(f"Classification task with {len(unique_vals)} classes")
        else:
            self.target_encoder = None
            logger.info("Regression task")
        
        # Split data - this is the critical step to prevent data leakage
        # For classification with too many classes or very imbalanced classes, don't use stratified split
        if self.task_type == 'classification':
            unique_vals = np.unique(y)
            if len(unique_vals) > 1000:  # Too many classes for stratified split
                logger.warning(f"Too many classes ({len(unique_vals)}) for stratified split, using random split")
                stratify = None
            else:
                # Check if any class has too few samples for stratified split
                class_counts = np.bincount(y)
                if np.any(class_counts < 2):
                    logger.warning("Some classes have too few samples for stratified split, using random split")
                    stratify = None
                else:
                    stratify = y
        else:
            stratify = None
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=stratify
        )
        # Keep raw splits for models that expect original DataFrames (e.g., FTTransformer)
        self.X_train_raw, self.X_test_raw = X_train.copy(), X_test.copy()
        
        # Preprocess data
        self.preprocessor = DataPreprocessor()
        X_train_processed, X_test_processed = self.preprocessor.fit_transform(X_train, X_test)
        
        self.X_train, self.X_test = X_train_processed, X_test_processed
        self.y_train, self.y_test = y_train, y_test
        
        logger.info(f"Training set: {self.X_train.shape}")
        logger.info(f"Test set: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_xgboost(self, **params):
        """Train XGBoost model"""
        logger.info("Training XGBoost...")
        
        # Create a small validation split from training for early stopping and fair benchmarking
        from sklearn.model_selection import train_test_split
        Xtr, Xval, ytr, yval = train_test_split(self.X_train, self.y_train, test_size=0.1, random_state=self.random_state, stratify=self.y_train if self.task_type=='classification' else None)

        if self.task_type == 'classification':
            model = xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss' if len(np.unique(self.y_train)) == 2 else 'mlogloss',
                **params
            )
            model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        else:
            model = xgb.XGBRegressor(
                random_state=self.random_state,
                eval_metric='rmse',
                **params
            )
            model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        self.models['XGBoost'] = model
        
        # Evaluate with appropriate metrics
        if self.task_type == 'classification':
            train_pred = model.predict(self.X_train)
            test_pred = model.predict(self.X_test)
            train_score = f1_score(self.y_train, train_pred, average='weighted')
            test_score = f1_score(self.y_test, test_pred, average='weighted')
        else:
            train_pred = model.predict(self.X_train)
            test_pred = model.predict(self.X_test)
            train_score = mean_squared_error(self.y_train, train_pred)
            test_score = mean_squared_error(self.y_test, test_pred)
        
        self.results['XGBoost'] = {
            'train_score': train_score,
            'test_score': test_score,
            'model': model
        }
        
        logger.info(f"XGBoost - Train: {train_score:.4f}, Test: {test_score:.4f}")
        return model
    
    def train_lightgbm(self, **params):
        """Train LightGBM model"""
        logger.info("Training LightGBM...")
        
        from sklearn.model_selection import train_test_split
        Xtr, Xval, ytr, yval = train_test_split(self.X_train, self.y_train, test_size=0.1, random_state=self.random_state, stratify=self.y_train if self.task_type=='classification' else None)

        if self.task_type == 'classification':
            model = lgb.LGBMClassifier(
                random_state=self.random_state,
                **params
            )
            model.fit(Xtr, ytr, eval_set=[(Xval, yval)])
        else:
            model = lgb.LGBMRegressor(
                random_state=self.random_state,
                **params
            )
            model.fit(Xtr, ytr, eval_set=[(Xval, yval)])
        self.models['LightGBM'] = model
        
        # Evaluate with appropriate metrics
        if self.task_type == 'classification':
            train_pred = model.predict(self.X_train)
            test_pred = model.predict(self.X_test)
            train_score = f1_score(self.y_train, train_pred, average='weighted')
            test_score = f1_score(self.y_test, test_pred, average='weighted')
        else:
            train_pred = model.predict(self.X_train)
            test_pred = model.predict(self.X_test)
            train_score = mean_squared_error(self.y_train, train_pred)
            test_score = mean_squared_error(self.y_test, test_pred)
        
        self.results['LightGBM'] = {
            'train_score': train_score,
            'test_score': test_score,
            'model': model
        }
        
        logger.info(f"LightGBM - Train: {train_score:.4f}, Test: {test_score:.4f}")
        return model
    
    def train_catboost(self, **params):
        """Train CatBoost model"""
        logger.info("Training CatBoost...")
        
        from sklearn.model_selection import train_test_split
        Xtr, Xval, ytr, yval = train_test_split(self.X_train, self.y_train, test_size=0.1, random_state=self.random_state, stratify=self.y_train if self.task_type=='classification' else None)

        if self.task_type == 'classification':
            model = CatBoostClassifier(
                random_state=self.random_state,
                verbose=False,
                **params
            )
            model.fit(Xtr, ytr, eval_set=(Xval, yval), verbose=False)
        else:
            model = CatBoostRegressor(
                random_state=self.random_state,
                verbose=False,
                **params
            )
            model.fit(Xtr, ytr, eval_set=(Xval, yval), verbose=False)
        self.models['CatBoost'] = model
        
        # Evaluate with appropriate metrics
        if self.task_type == 'classification':
            train_pred = model.predict(self.X_train)
            test_pred = model.predict(self.X_test)
            train_score = f1_score(self.y_train, train_pred, average='weighted')
            test_score = f1_score(self.y_test, test_pred, average='weighted')
        else:
            train_pred = model.predict(self.X_train)
            test_pred = model.predict(self.X_test)
            train_score = mean_squared_error(self.y_train, train_pred)
            test_score = mean_squared_error(self.y_test, test_pred)
        
        self.results['CatBoost'] = {
            'train_score': train_score,
            'test_score': test_score,
            'model': model
        }
        
        logger.info(f"CatBoost - Train: {train_score:.4f}, Test: {test_score:.4f}")
        return model
    
    def train_mlp(self, hidden_dims=[256, 128, 64], epochs=100, lr=0.001, batch_size=32, **params):
        """Train MLP model"""
        logger.info("Training MLP...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Prepare data
        train_dataset = MLPDataset(self.X_train, self.y_train, self.task_type)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Model
        input_dim = self.X_train.shape[1]
        output_dim = len(np.unique(self.y_train)) if self.task_type == 'classification' else 1
        
        model = MLP(input_dim, hidden_dims, output_dim, **params).to(device)
        
        # Loss and optimizer
        if self.task_type == 'classification':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                
                if self.task_type == 'regression':
                    outputs = outputs.squeeze()
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
        
        self.models['MLP'] = model
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            X_train_tensor = torch.FloatTensor(self.X_train).to(device)
            X_test_tensor = torch.FloatTensor(self.X_test).to(device)
            
            train_pred = model(X_train_tensor)
            test_pred = model(X_test_tensor)
            
            if self.task_type == 'classification':
                train_pred = torch.argmax(train_pred, dim=1).cpu().numpy()
                test_pred = torch.argmax(test_pred, dim=1).cpu().numpy()
            else:
                train_pred = train_pred.squeeze().cpu().numpy()
                test_pred = test_pred.squeeze().cpu().numpy()
        
        if self.task_type == 'classification':
            train_score = f1_score(self.y_train, train_pred, average='weighted')
            test_score = f1_score(self.y_test, test_pred, average='weighted')
        else:
            train_score = mean_squared_error(self.y_train, train_pred)
            test_score = mean_squared_error(self.y_test, test_pred)
        
        self.results['MLP'] = {
            'train_score': train_score,
            'test_score': test_score,
            'model': model
        }
        
        logger.info(f"MLP - Train: {train_score:.4f}, Test: {test_score:.4f}")
        return model
    
    def train_tabnet(self, epochs=100, batch_size=256, **params):
        """Train TabNet model"""
        if not TABNET_AVAILABLE:
            logger.warning("TabNet not available, skipping...")
            return None
        
        logger.info("Training TabNet...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.task_type == 'classification':
            model = TabNetClassifier(
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2),
                scheduler_params={"step_size":50, "gamma":0.9},
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                mask_type='entmax',
                **params
            )
        else:
            model = TabNetRegressor(
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2),
                scheduler_params={"step_size":50, "gamma":0.9},
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                mask_type='entmax',
                **params
            )
        
        # Train
        # Create a validation split from training data to avoid using test labels for early stopping
        from sklearn.model_selection import train_test_split
        Xtr, Xval, ytr, yval = train_test_split(self.X_train, self.y_train, test_size=0.1, random_state=42, stratify=self.y_train if self.task_type=='classification' else None)
        model.fit(
            X_train=Xtr, y_train=ytr,
            eval_set=[(Xval, yval)],
            max_epochs=epochs,
            patience=20,
            batch_size=batch_size,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )
        
        self.models['TabNet'] = model
        
        # Evaluate
        train_pred = model.predict(self.X_train)
        test_pred = model.predict(self.X_test)
        
        if self.task_type == 'classification':
            train_score = f1_score(self.y_train, train_pred, average='weighted')
            test_score = f1_score(self.y_test, test_pred, average='weighted')
        else:
            train_score = mean_squared_error(self.y_train, train_pred)
            test_score = mean_squared_error(self.y_test, test_pred)
        
        self.results['TabNet'] = {
            'train_score': train_score,
            'test_score': test_score,
            'model': model
        }
        
        logger.info(f"TabNet - Train: {train_score:.4f}, Test: {test_score:.4f}")
        return model
    
    def train_fttransformer(self, epochs=100, batch_size=256, **params):
        """Train FTTransformer model"""
        if not FTTRANSFORMER_AVAILABLE:
            logger.warning("FTTransformer not available, skipping...")
            return None
        
        logger.info("Training FTTransformer...")
        
        # Prepare data for FTTransformer using raw DataFrames and original column names
        cat_cols = self.preprocessor.categorical_columns
        cont_cols = self.preprocessor.numerical_columns

        # Build train/val split from raw training data
        from sklearn.model_selection import train_test_split
        Xtr_raw, Xval_raw, ytr, yval = train_test_split(self.X_train_raw, self.y_train, test_size=0.1, random_state=42, stratify=self.y_train if self.task_type=='classification' else None)

        # Create dataframes with original columns
        train_df = Xtr_raw.copy()
        val_df = Xval_raw.copy()
        train_df['target'] = ytr
        val_df['target'] = yval
        
        # Model configuration
        model_config = ModelConfig(
            task="classification" if self.task_type == 'classification' else "regression",
            categorical_cols=cat_cols,
            continuous_cols=cont_cols,
            target_col=['target'],
            embedding_dims=[[2] * len(cat_cols)] if cat_cols else None,
            continuous_embedding_dim=128,
            depth=4,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
            **params
        )
        
        optimizer_config = OptimizerConfig(
            optimizer="AdamW",
            lr=1e-3,
            weight_decay=1e-5
        )
        
        trainer_config = TrainerConfig(
            max_epochs=epochs,
            batch_size=batch_size,
            early_stopping="valid_loss",
            early_stopping_patience=10,
            checkpoints="best",
            load_best=True
        )
        
        # Create and train model
        model = TabularModel(
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config
        )
        
        model.fit(train=train_df, validation=val_df)
        
        self.models['FTTransformer'] = model
        
        # Evaluate
        train_pred = model.predict(train_df)
        test_pred = model.predict(pd.concat([self.X_test_raw.copy(), pd.Series(self.y_test, name='target')], axis=1))
        
        if self.task_type == 'classification':
            train_score = f1_score(self.y_train, train_pred['target_prediction'], average='weighted')
            test_score = f1_score(self.y_test, test_pred['target_prediction'], average='weighted')
        else:
            train_score = mean_squared_error(self.y_train, train_pred['target_prediction'])
            test_score = mean_squared_error(self.y_test, test_pred['target_prediction'])
        
        self.results['FTTransformer'] = {
            'train_score': train_score,
            'test_score': test_score,
            'model': model
        }
        
        logger.info(f"FTTransformer - Train: {train_score:.4f}, Test: {test_score:.4f}")
        return model
    
    def train_autogluon(self, time_limit=300, **params):
        """Train AutoGluon model"""
        if not AUTOGLUON_AVAILABLE:
            logger.warning("AutoGluon not available, skipping...")
            return None
        
        logger.info("Training AutoGluon...")
        
        # Prepare data for AutoGluon
        train_df = pd.DataFrame(self.X_train)
        test_df = pd.DataFrame(self.X_test)
        
        # Add column names
        train_df.columns = [f'feature_{i}' for i in range(train_df.shape[1])]
        test_df.columns = [f'feature_{i}' for i in range(test_df.shape[1])]
        
        # Add target
        train_df['target'] = self.y_train
        test_df['target'] = self.y_test
        
        # Create predictor
        predictor = TabularPredictor(
            label='target',
            eval_metric='accuracy' if self.task_type == 'classification' else 'root_mean_squared_error',
            **params
        )
        
        # Train
        predictor.fit(
            train_data=train_df,
            time_limit=time_limit,
            presets='best_quality'
        )
        
        self.models['AutoGluon'] = predictor
        
        # Evaluate
        train_pred = predictor.predict(train_df)
        test_pred = predictor.predict(test_df)
        
        if self.task_type == 'classification':
            train_score = f1_score(self.y_train, train_pred, average='weighted')
            test_score = f1_score(self.y_test, test_pred, average='weighted')
        else:
            train_score = mean_squared_error(self.y_train, train_pred)
            test_score = mean_squared_error(self.y_test, test_pred)
        
        self.results['AutoGluon'] = {
            'train_score': train_score,
            'test_score': test_score,
            'model': predictor
        }
        
        logger.info(f"AutoGluon - Train: {train_score:.4f}, Test: {test_score:.4f}")
        return predictor
    
    def train_all_models(self, **kwargs):
        """Train all available models"""
        logger.info("Starting training of all models...")
        
        # Train traditional ML models
        self.train_xgboost(**kwargs.get('xgboost', {}))
        self.train_lightgbm(**kwargs.get('lightgbm', {}))
        self.train_catboost(**kwargs.get('catboost', {}))
        
        # Train deep learning models
        self.train_mlp(**kwargs.get('mlp', {}))
        
        if TABNET_AVAILABLE:
            self.train_tabnet(**kwargs.get('tabnet', {}))
        
        if FTTRANSFORMER_AVAILABLE:
            self.train_fttransformer(**kwargs.get('fttransformer', {}))
        
        # AutoGluon is disabled by default due to long training time
        # Uncomment the following lines if you want to include AutoGluon
        # if AUTOGLUON_AVAILABLE:
        #     self.train_autogluon(**kwargs.get('autogluon', {}))
        
        logger.info("All models trained successfully!")
        return self.results
    
    def get_best_model(self):
        """Get the best performing model based on test score"""
        if not self.results:
            return None
        
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['test_score'])
        return best_model_name, self.results[best_model_name]
    
    def plot_results(self, save_path=None):
        """Plot comparison of model results"""
        if not self.results:
            logger.warning("No results to plot")
            return
        
        models = list(self.results.keys())
        train_scores = [self.results[model]['train_score'] for model in models]
        test_scores = [self.results[model]['test_score'] for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Train vs Test scores
        x = np.arange(len(models))
        width = 0.35
        
        ax1.bar(x - width/2, train_scores, width, label='Train Score', alpha=0.8)
        ax1.bar(x + width/2, test_scores, width, label='Test Score', alpha=0.8)
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Train vs Test Scores')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Test scores only
        ax2.bar(models, test_scores, alpha=0.8, color='skyblue')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Test Score')
        ax2.set_title('Test Scores Comparison')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_results_to_csv(self, dataset_name, column_name=None):
        """Save results to results_model.csv file, robustly updating the correct row and column"""
        import pandas as pd
        csv_file = "results_model.csv"

        # Read existing CSV file
        try:
            results_df = pd.read_csv(csv_file)
        except FileNotFoundError:
            # Create new DataFrame if file doesn't exist
            results_df = pd.DataFrame(columns=['Model'])

        # Clean up column names (strip whitespace)
        results_df.columns = [c.strip() for c in results_df.columns]
        if 'Model' not in results_df.columns:
            results_df.insert(0, 'Model', '')

        # Clean up model names (strip whitespace)
        results_df['Model'] = results_df['Model'].astype(str).str.strip()

        # Determine which column to update
        if column_name is None:
            column_name = dataset_name.replace('.csv', '').replace('.data', '')
        column_name = column_name.strip()

        # Add column if it doesn't exist
        if column_name not in results_df.columns:
            results_df[column_name] = None

        # List of all models (to preserve order)
        all_models = [
            'XGBoost', 'LightGBM', 'CatBoost', 'MLP', 'TabNet', 'FTTransformer', 'AutoGluon', 'SBLN'
        ]

        # Ensure all model rows exist (in correct order)
        for model in all_models:
            if not (results_df['Model'] == model).any():
                results_df = pd.concat([
                    results_df,
                    pd.DataFrame({'Model': [model]})
                ], ignore_index=True)

        # Update results for each model
        for model_name, result in self.results.items():
            model_name_clean = model_name.strip()
            mask = results_df['Model'] == model_name_clean
            if mask.any():
                results_df.loc[mask, column_name] = result['test_score']
            else:
                # Should not happen, but just in case
                new_row = {col: None for col in results_df.columns}
                new_row['Model'] = model_name_clean
                new_row[column_name] = result['test_score']
                results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

        # Reorder rows to match all_models
        results_df['Model'] = results_df['Model'].astype(str).str.strip()
        results_df = results_df.set_index('Model').reindex(all_models).reset_index()

        # Save updated CSV
        results_df.to_csv(csv_file, index=False)
        logger.info(f"Results saved to {csv_file} in column '{column_name}'")
        return results_df
    
    def save_results(self, filepath):
        """Save results to JSON file"""
        # Convert results to serializable format
        serializable_results = {}
        for model_name, result in self.results.items():
            serializable_results[model_name] = {
                'train_score': float(result['train_score']),
                'test_score': float(result['test_score']),
                'model_type': type(result['model']).__name__
            }
        
        # Add metadata
        metadata = {
            'task_type': self.task_type,
            'data_shape': self.X_train.shape,
            'timestamp': datetime.now().isoformat(),
            'results': serializable_results
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")

def main():
    """Main function to run the training script"""
    
    # Configuration
    DATASET_PATH = "data/Iris.csv"  # Change this to your preferred dataset
    TARGET_COLUMN = "Species"  # Change this to your target column
    COLUMN_NAME = "Iris"  # Column name in results_model.csv
    
    # Available datasets and their target columns
    DATASET_CONFIGS = {
        "Iris.csv": "Species",
        "breast_cancer.csv": "diagnosis",  # Assuming this column exists
        "adult.csv": "income",  # Assuming this column exists
        "train.csv": "target",  # Assuming this column exists
        "test_data9.csv": "target",  # Assuming this column exists
    }
    
    # Model parameters
    model_params = {
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1
        },
        'lightgbm': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1
        },
        'catboost': {
            'iterations': 100,
            'depth': 6,
            'learning_rate': 0.1
        },
        'mlp': {
            'hidden_dims': [256, 128, 64],
            'epochs': 50,
            'lr': 0.001,
            'batch_size': 64
        },
        'tabnet': {
            'epochs': 50,
            'batch_size': 256
        },
        'fttransformer': {
            'epochs': 50,
            'batch_size': 256
        },
        'autogluon': {
            'time_limit': 300
        }
    }
    
    # Initialize trainer
    trainer = ModelTrainer(random_state=42)
    
    # Load and preprocess data
    logger.info(f"Using dataset: {DATASET_PATH}")
    trainer.load_data(DATASET_PATH, TARGET_COLUMN, test_size=0.2)
    
    # Train all models
    results = trainer.train_all_models(**model_params)
    
    # Get best model
    best_model_name, best_result = trainer.get_best_model()
    logger.info(f"Best model: {best_model_name} with test score: {best_result['test_score']:.4f}")
    
    # Save results to CSV
    trainer.save_results_to_csv(DATASET_PATH, COLUMN_NAME)
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    for model_name, result in results.items():
        print(f"{model_name:15} | Train: {result['train_score']:.4f} | Test: {result['test_score']:.4f}")
    print("="*50)
    print(f"Best Model: {best_model_name}")
    print(f"Best Test Score: {best_result['test_score']:.4f}")
    print(f"Results saved to results_model.csv in column '{COLUMN_NAME}'")

if __name__ == "__main__":
    main() 