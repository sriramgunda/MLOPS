# Automated ML Pipeline Implementation Plan

## Goal
Build an automated machine learning pipeline for the Heart Disease dataset using FLAML and MLflow.

## Status: Completed

## Components

### 1. Data Loading & Preprocessing
- **File:** `src/data_loader.py` (Implicitly handled in `train_automl.py` for now)
- **Status:** Implemented.
- **Details:** Loads `data/heart.csv`, performs OHE and Scaling via Scikit-Learn Pipeline.

### 2. Exploratory Data Analysis (EDA)
- **File:** `src/auto_eda.py`
- **Status:** Implemented.
- **Details:**
  - Generates Sweetviz report: `plots/sweetviz_report.html`
  - Generates Pandas Profiling report: `plots/pandas_profiling_report.html`

### 3. Model Training (AutoML)
- **File:** `src/train_automl.py`
- **Status:** Implemented.
- **Details:**
  - Uses FLAML for automated model selection and hyperparameter tuning.
  - Time budget: 60 seconds.
  - Metric: Custom metric trying to optimize `1 - ROC_AUC` while tracking Accuracy, Precision, Recall, F1.

### 4. Experiment Tracking (MLflow)
- **Status:** Implemented.
- **Details:**
  - Experiment Name: `Heart_Disease_AutoML`
  - Logs every FLAML trial as a nested run.
  - Logs hyperparameters and metrics for each trial.
  - Logs human-readable `model_type` (e.g., "Random Forest" instead of "rf").

### 5. Model Artifacts
- **Status:** Implemented.
- **Details:**
  - Best model saved as `best_model.pkl`.
  - Best model logged to MLflow artifacts.
  - Best config saved as `best_config.json`.

## Usage
1. **Run EDA:**
   ```bash
   python src/auto_eda.py
   ```
2. **Run Training Pipeline:**
   ```bash
   python src/train_automl.py
   ```
3. **View MLflow Dashboard:**
   ```bash
   mlflow ui
   ```
