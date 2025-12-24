# Heart Disease AutoML Pipeline

This project implements a robust MLOps pipeline for predicting heart disease using the UCI Heart Disease dataset. It leverages **FLAML** for automated model selection and hyperparameter tuning, and **MLflow** for comprehensive experiment tracking.

## 🚀 Features
- **Automated Data Acquisition:** Automatically downloads the Cleveland heart disease dataset with local fallback.
- **Auto-EDA:** Generates a detailed Pandas Profiling report.
- **AutoML with FLAML:** Explores multiple models (Random Forest, XGBoost, LGBM, Logistic Regression) to find the best performer.
- **Experiment Tracking:** Logs parameters, metrics (Accuracy, ROC-AUC, F1), and artifacts to **MLflow**.
- **Full Reproducibility:** Saves a unified **Inference Pipeline** (Preprocessor + Model) that accepts raw data.
- **Automated Reporting:** Generates a `tuning_report.md` summarizing the search process and per-model leaderboard.

## 📁 Project Structure
- `src/`:
  - `data_loader.py`: Handles data download and cleaning.
  - `auto_eda.py`: Generates the EDA profiling report.
  - `train_automl.py`: Executes the AutoML tuning and MLflow logging.
  - `predict.py`: Demonstrates inference by loading the best model from MLflow.
- `plots/`: Contains the `pandas_profiling_report.html`.
- `requirements.txt`: Project dependencies.
- `mlruns/`: Local MLflow tracking database.

## 🛠️ Setup
Install the dependencies:
```bash
pip install -r requirements.txt
```

## 🔄 Running the Pipeline

### 1. Unified Run (EDA + Training)
Execute the complete pipeline in one command:
```bash
python src/auto_eda.py && python src/train_automl.py
```

### 2. Verify with Inference
Run the prediction script to load the latest best model from MLflow and test it on raw data:
```bash
python src/predict.py
```

### 3. Explore Experiments
Launch the MLflow UI to view model search history, tuning reports, and performance plots:
```bash
mlflow ui
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

## 📊 Outputs & Artifacts
Every training run produces:
- `inference_pipeline.pkl`: The complete fitted pipeline for production.
- `tuning_report.md`: Detailed breakdown of the hyperparameter search.
- `model_selection_reasoning.txt`: Plain-text explanation of why the best model was chosen.
- `roc_curve.png` & `confusion_matrix.png`: Performance visualizations.
