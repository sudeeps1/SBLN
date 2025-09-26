#!/usr/bin/env python3
"""
Simple script to train models on a chosen dataset and save results to results_model.csv
"""

import os
import sys
from multi_model_training import ModelTrainer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def list_available_datasets():
    """List all available CSV datasets in the data directory"""
    data_dir = "data"
    datasets = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith('.csv') or file.endswith('.data'):
                datasets.append(file)
    
    return sorted(datasets)

def get_dataset_info(file_path):
    """Get basic information about a dataset"""
    import pandas as pd
    
    try:
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        else:
            # For .data files, check if first row looks like header
            data = pd.read_csv(file_path, sep='\s+', header=None)
            
            # Check if first row contains header-like values
            first_row = data.iloc[0].astype(str)
            if any('class' in val.lower() or 'id' in val.lower() or 'sequence' in val.lower() for val in first_row):
                print("Detected header row, skipping it...")
                data = pd.read_csv(file_path, sep='\s+', header=0)
        
        print(f"\nDataset: {file_path}")
        print(f"Shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        # Show sample of target column (last column)
        target_col = data.columns[-1]
        print(f"\nTarget column: {target_col}")
        print(f"Target values: {data[target_col].value_counts().head()}")
        
        # Provide suggestions for problematic datasets
        if file_path.endswith('splice.data'):
            print(f"\n⚠️  WARNING: Column 2 has {data[data.columns[2]].nunique()} unique sequences!")
            print("💡 SUGGESTION: Use column 0 (class labels) instead - it has only 4 unique values")
            print("   Column 0 values:", data[data.columns[0]].unique())
        elif file_path.endswith('DNAsequence.csv'):
            print(f"\n⚠️  WARNING: NucleotideSequence has {data['NucleotideSequence'].nunique()} unique sequences!")
            print("💡 SUGGESTION: Use 'GeneType' instead - it has only 10 unique values")
            print("   GeneType values:", data['GeneType'].unique())
        
        return data.columns[-1]  # Return last column as target
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def main():
    """Main function to run training with dataset selection"""
    
    print("="*60)
    print("MODEL TRAINING SCRIPT")
    print("="*60)
    
    # List available datasets
    datasets = list_available_datasets()
    
    if not datasets:
        print("No CSV or .data files found in data/ directory!")
        return
    
    print("\nAvailable datasets:")
    for i, dataset in enumerate(datasets, 1):
        print(f"{i}. {dataset}")
    
    # Get user choice
    while True:
        try:
            choice = input(f"\nSelect dataset (1-{len(datasets)}): ").strip()
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(datasets):
                selected_dataset = datasets[choice_idx]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Show dataset info
    dataset_path = f"data/{selected_dataset}"
    target_column = get_dataset_info(dataset_path)
    
    if target_column is None:
        print("Could not read dataset. Exiting.")
        return
    
    # Ask for custom target column
    custom_target = input(f"\nUse '{target_column}' as target column? (y/n): ").strip().lower()
    if custom_target != 'y':
        target_column = input("Enter target column name: ").strip()
    
    # Ask for column name in results CSV
    column_name = input(f"\nEnter column name for results in results_model.csv (or press Enter for '{selected_dataset.replace('.csv', '').replace('.data', '')}'): ").strip()
    if not column_name:
        column_name = selected_dataset.replace('.csv', '').replace('.data', '')
    
    # Configuration
    print(f"\nUsing dataset: {dataset_path}")
    print(f"Target column: {target_column}")
    print(f"Results column: {column_name}")
    
    # Model parameters (optimized for speed)
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
            'batch_size': 32
        },
        'tabnet': {
            'epochs': 50,
            'batch_size': 256
        },
        'fttransformer': {
            'epochs': 50,
            'batch_size': 256
        }
    }
    
    # Initialize trainer
    trainer = ModelTrainer(random_state=42)
    
    try:
        # Load and preprocess data
        logger.info(f"Loading dataset: {dataset_path}")
        # Optional: ask for columns to exclude to prevent leakage
        excl = input("\nColumns to exclude from features (comma-separated, or Enter to skip): ").strip()
        exclude_columns = [c.strip() for c in excl.split(',') if c.strip()] if excl else []

        trainer.load_data(dataset_path, target_column, test_size=0.2, exclude_columns=exclude_columns)
        
        # Train all models
        logger.info("Starting model training...")
        results = trainer.train_all_models(**model_params)
        
        # Get best model
        best_model_name, best_result = trainer.get_best_model()
        logger.info(f"Best model: {best_model_name} with test score: {best_result['test_score']:.4f}")
        
        # Save results to CSV
        trainer.save_results_to_csv(selected_dataset, column_name)
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        for model_name, result in results.items():
            print(f"{model_name:15} | Train: {result['train_score']:.4f} | Test: {result['test_score']:.4f}")
        print("="*60)
        print(f"Best Model: {best_model_name}")
        print(f"Best Test Score: {best_result['test_score']:.4f}")
        print(f"\nResults saved to results_model.csv in column '{column_name}'")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        print(f"\nError: {e}")
        print("Please check your dataset and try again.")

if __name__ == "__main__":
    main() 