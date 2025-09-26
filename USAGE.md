# Model Training Usage Guide

## Quick Start

### Option 1: Interactive Training (Recommended)
```bash
python train_model.py
```
This will:
1. Show you all available datasets in the `data/` directory
2. Let you choose which dataset to train on
3. Let you specify the target column
4. Let you choose which column name to use in `results_model.csv`
5. Train all models and save results

### Option 2: Direct Training
Edit the configuration in `multi_model_training.py`:
```python
DATASET_PATH = "data/your_dataset.csv"  # Change this
TARGET_COLUMN = "your_target_column"    # Change this  
COLUMN_NAME = "YourColumnName"          # Change this
```
Then run:
```bash
python multi_model_training.py
```

## Available Datasets

The following datasets are available in the `data/` directory:
- `Iris.csv` - Iris flower classification
- `breast_cancer.csv` - Breast cancer diagnosis
- `adult.csv` - Adult income prediction
- `titanic.csv` - Titanic survival prediction
- `aneurysm.csv` - Aneurysm prediction
- `polymer.csv` - Polymer properties
- `parkinsons_updrs.data` - Parkinson's disease
- `splice.data` - DNA splice site prediction
- `higgs.csv` - Higgs boson detection
- `mnist_train.csv` - MNIST digit recognition
- `random.csv` - Random dataset

## Results Output

Results are automatically saved to `results_model.csv` with:
- **F1 Score** for classification tasks
- **MSE (Mean Squared Error)** for regression tasks

The CSV file structure:
```
Model, Adult, Aneurysm, Breast Cancer, Higgs, Iris, mnist, Parkinsons, Polymer, Splice, Titanic
XGBoost, 0.85, 0.92, 0.95, 0.78, 1.0, 0.98, 0.88, 0.91, 0.87, 0.82
LightGBM, 0.84, 0.91, 0.94, 0.77, 1.0, 0.97, 0.87, 0.90, 0.86, 0.81
CatBoost, 0.86, 0.93, 0.96, 0.79, 1.0, 0.99, 0.89, 0.92, 0.88, 0.83
MLP, 0.83, 0.90, 0.93, 0.76, 1.0, 0.96, 0.86, 0.89, 0.85, 0.80
TabNet, 0.82, 0.89, 0.92, 0.75, 0.97, 0.95, 0.85, 0.88, 0.84, 0.79
```

## Models Trained

The script trains the following models:
1. **XGBoost** - Gradient boosting
2. **LightGBM** - Light gradient boosting machine
3. **CatBoost** - Categorical boosting
4. **MLP** - Multi-layer perceptron (PyTorch)
5. **TabNet** - Attention-based tabular learning (if available)

Note: AutoGluon and FTTransformer are disabled by default due to long training times.

## Data Leakage Prevention

The script ensures no data leakage by:
- ✅ Proper train/test split before preprocessing
- ✅ Preprocessors fitted only on training data
- ✅ Test data transformed using fitted preprocessors
- ✅ Stratified sampling for classification tasks

## Example Usage

```bash
# Run interactive training
python train_model.py

# Select dataset: 1 (Iris.csv)
# Use target column: Species
# Column name: Iris

# Results will be saved to results_model.csv in the "Iris" column
```

## Troubleshooting

1. **Missing dependencies**: Run `pip install -r requirements.txt`
2. **Dataset not found**: Make sure your dataset is in the `data/` directory
3. **Column not found**: Check the column names in your dataset
4. **Memory issues**: Reduce batch sizes or epochs in model parameters 