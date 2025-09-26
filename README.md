# Multi-Model Training Script

This repository contains comprehensive scripts to train multiple machine learning models on tabular datasets while ensuring **no data leakage**. The scripts support both classification and regression tasks.

## Features

### Models Supported
- **XGBoost** - Gradient boosting with XGBoost
- **LightGBM** - Light gradient boosting machine
- **CatBoost** - Categorical boosting
- **MLP** - Multi-layer perceptron (PyTorch)
- **TabNet** - Attention-based tabular learning
- **FTTransformer** - Feature tokenizer transformer
- **AutoGluon** - AutoML framework

### Data Leakage Prevention
- ✅ Proper train/test split before any preprocessing
- ✅ Preprocessors fitted only on training data
- ✅ Test data transformed using fitted preprocessors
- ✅ No information from test set used during training
- ✅ Stratified sampling for classification tasks

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Optional: Install additional packages for specific models:**
```bash
# For TabNet
pip install pytorch-tabnet

# For FTTransformer
pip install pytorch-tabular

# For AutoGluon
pip install autogluon.tabular
```

## Usage

### Quick Start

1. **Interactive dataset selection:**
```bash
python run_training.py
```
This will:
- List all available CSV/.data files
- Let you choose a dataset
- Show dataset information
- Train all available models
- Generate comparison plots and save results

2. **Direct execution with specific dataset:**
```bash
python multi_model_training.py
```
Edit the `DATASET_PATH` and `TARGET_COLUMN` variables in the script.

### Custom Usage

```python
from multi_model_training import ModelTrainer

# Initialize trainer
trainer = ModelTrainer(random_state=42)

# Load and preprocess data
trainer.load_data("your_dataset.csv", "target_column", test_size=0.2)

# Train all models
results = trainer.train_all_models()

# Get best model
best_model_name, best_result = trainer.get_best_model()
print(f"Best model: {best_model_name}")

# Plot results
trainer.plot_results(save_path="comparison.png")

# Save results
trainer.save_results("results.json")
```

## Data Format

The scripts support:
- **CSV files** (`.csv`)
- **Space/tab-separated files** (`.data`)

### Expected Format
- Features in columns (except target)
- Target column specified by name
- Automatic detection of classification vs regression

## Model Configuration

You can customize model parameters:

```python
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
    },
    'autogluon': {
        'time_limit': 300
    }
}
```

## Output

The scripts generate:

1. **Training Results** - JSON file with all model scores
2. **Comparison Plot** - Visual comparison of model performance
3. **Console Output** - Detailed training progress and final summary

### Example Output
```
==================================================
TRAINING SUMMARY
==================================================
XGBoost        | Train: 0.9800 | Test: 0.9500
LightGBM       | Train: 0.9750 | Test: 0.9400
CatBoost       | Train: 0.9700 | Test: 0.9450
MLP            | Train: 0.9650 | Test: 0.9350
TabNet         | Train: 0.9600 | Test: 0.9300
==================================================
Best Model: XGBoost
Best Test Score: 0.9500
```

## Data Preprocessing

The `DataPreprocessor` class handles:

- **Numerical features**: Median imputation + Robust scaling
- **Categorical features**: Constant imputation + Label encoding
- **Train/test separation**: Preprocessors fitted only on training data
- **Automatic column detection**: Based on data types

## Available Datasets

The scripts can work with any of your CSV files. Some examples in your workspace:
- `Iris.csv` - Classic iris classification
- `breast_cancer.csv` - Breast cancer diagnosis
- `adult.csv` - Adult income prediction
- `train.csv` - Generic training data
- `test_data9.csv` - Test dataset

## Troubleshooting

### Common Issues

1. **Missing dependencies:**
   - Install required packages: `pip install -r requirements.txt`
   - Some models (TabNet, FTTransformer, AutoGluon) are optional

2. **Memory issues:**
   - Reduce batch sizes for deep learning models
   - Use smaller datasets for testing

3. **CUDA issues:**
   - Models automatically fall back to CPU if CUDA unavailable
   - Check PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`

4. **Dataset format issues:**
   - Ensure target column exists in dataset
   - Check for proper CSV formatting

### Performance Tips

- **For large datasets**: Reduce epochs and batch sizes
- **For faster training**: Use fewer estimators for gradient boosting
- **For better results**: Increase training time for AutoGluon

## File Structure

```
├── multi_model_training.py    # Main training script
├── run_training.py           # Interactive dataset selection
├── requirements.txt          # Dependencies
├── README.md                # This file
├── *.csv                    # Your datasets
└── results/                 # Generated outputs
    ├── training_results_*.json
    └── model_comparison_*.png
```

## Contributing

Feel free to:
- Add new models
- Improve preprocessing
- Enhance visualization
- Add cross-validation
- Implement hyperparameter tuning

## License

This project is open source and available under the MIT License. 