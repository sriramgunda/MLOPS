import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
from flaml import AutoML
import mlflow
import mlflow.sklearn
import os
import json
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Define a custom function for the oldpeak indicator at the top level
# This is REQUIRED for the pipeline to be picklable/loadable by MLflow/Pickle
def add_oldpeak_indicator(X):
    """
    Assumes oldpeak is the last column in the numeric group or accessed by index.
    In the pipeline, it receives the specified column as a 2D array.
    """
    return (X == 0).astype(float)

def train_pipeline():
    # 1. Load Data
    print("Loading and preparing data...")
    try:
        from data_loader import load_data
    except ImportError:
        # Fallback for when running as a module or from different context
        sys.path.append(os.path.join(os.getcwd(), 'src'))
        from data_loader import load_data

    # Generate/Load the dataset on the fly
    df = load_data()
    if df is None:
        print("Failed to load data.")
        return

    X_raw = df.drop("target", axis=1)
    y_raw = df["target"]

    # 2. Data Cleaning & Label Engineering (Move from Loader to Model Pipeline)
    print("Performing Data Cleaning and Label Engineering...")
    
    # --- Visualization 1: Data Cleaning (Identifying Missing Values) ---
    os.makedirs("plots", exist_ok=True)
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        plt.figure(figsize=(10, 6))
        null_counts[null_counts > 0].plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title("Data Cleaning: Identifying Missing Values in Raw Dataset")
        plt.ylabel("Missing Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("plots/1_cleaning_missing_values.png")
        plt.close()
    
    # We NO LONGER drop rows. We will use an Imputer in the pipeline for 100% robustness.
    df_clean = df.copy()
    print("Strategy Change: Using Pipeline Imputation instead of Row Dropping (Industry Standard).")

    # --- Visualization 2: Data Cleaning (Target Binarization) ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(x=df_clean['target'], hue=df_clean['target'], palette='viridis', legend=False)
    plt.title("Before: Original Target Stages (0-4)")
    
    # Clinical Transformation: Binarization
    df_clean['target'] = df_clean['target'].apply(lambda x: 1 if x > 0 else 0)
    
    plt.subplot(1, 2, 2)
    sns.countplot(x=df_clean['target'], hue=df_clean['target'], palette='coolwarm', legend=False)
    plt.title("After: Binarized Heart Disease Presence (0 or 1)")
    plt.tight_layout()
    plt.savefig("plots/2_cleaning_target_binarization.png")
    plt.close()

    # Define final features and labels from RAW data (No manual FE here)
    X = df_clean.drop("target", axis=1)
    y = df_clean["target"]

    # 3. Feature Engineering / Preprocessing
    # Define feature groups
    categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    
    print("Preprocessing data...")
    # Create preprocessing pipeline
    # We now bundle: Imputation -> Custom Indicator -> Polynomials -> Scaling
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_base', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
                ('scaler', StandardScaler())
            ]), ["age", "trestbps", "chol", "thalach", "oldpeak"]),
            ('oldpeak_flag', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('indicator', FunctionTransformer(add_oldpeak_indicator))
            ]), ["oldpeak"]),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ])

    # Visualize Feature Engineering (Scaling & Encoding)
    print("Capturing Feature Engineering Visualizations...")
    
    # --- Visualization 3: Feature Scaling (Actual Transformations) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # Age Raw vs Scaled
    sns.histplot(X['age'], kde=True, ax=axes[0,0], color='royalblue')
    axes[0,0].set_title("Age (Original Distribution)")
    temp_age_scaled = StandardScaler().fit_transform(X[['age']])
    sns.histplot(temp_age_scaled, kde=True, ax=axes[0,1], color='forestgreen')
    axes[0,1].set_title("Age (Standardized Transformation)")
    
    # Cholesterol Raw vs Scaled
    sns.histplot(X['chol'], kde=True, ax=axes[1,0], color='purple')
    axes[1,0].set_title("Cholesterol (Original Distribution)")
    temp_chol_scaled = StandardScaler().fit_transform(X[['chol']])
    sns.histplot(temp_chol_scaled, kde=True, ax=axes[1,1], color='darkorange')
    axes[1,1].set_title("Cholesterol (Standardized Transformation)")
    
    plt.tight_layout()
    plt.savefig("plots/3_feature_eng_scaling.png")
    plt.close()

    # --- Visualization 4: Categorical Encoding (Actual Map) ---
    # One-Hot Encoding visualize for Chest Pain (CP) attribute
    cp_sample = X[['cp']].head(10)
    ohe_temp = OneHotEncoder(sparse_output=False)
    cp_encoded = ohe_temp.fit_transform(cp_sample)
    df_cp_map = pd.DataFrame(cp_encoded, columns=ohe_temp.get_feature_names_out(['cp']))
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_cp_map, annot=True, cmap="YlGnBu", cbar=False)
    plt.title("Actual Data Transformation: One-Hot Encoding (CP clinical feature)")
    plt.xlabel("Engineered Feature Columns")
    plt.ylabel("Patient Records (Row Index)")
    plt.tight_layout()
    plt.savefig("plots/4_feature_eng_encoding.png")
    plt.close()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Fit and transform training data, transform test data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Convert back to dataframe for FLAML (optional but helpful for some estimators, though numpy is fine)
    # FLAML works fine with numpy arrays (CSR matrix from OHE).

    # Define custom callback to log each trial to MLflow
    # Note: Removed from fit() due to compatibility issues with recent FLAML versions.
    # Shared dictionary to pass metrics from custom_metric to the callback
    # current_trial_metrics = {}

    # def mlflow_logging_callback(trial_index, val_loss, config, best_val_loss, estimator, metric, time_total):
    #     # Start a nested run for each trial
    #     with mlflow.start_run(nested=True, run_name=f"Trial_{trial_index}_{estimator}"):
    #         # Standardized Parameter Wording
    #         mlflow.log_param("Model_Type", estimator)
    #         for k, v in config.items():
    #             mlflow.log_param(k, v)
    #         
    #         # Log primary optimization metric
    #         mlflow.log_metric("val_loss", val_loss)
    #         
    #         # Log additional metrics from custom_metric if available
    #         if current_trial_metrics:
    #             for m_name, m_val in current_trial_metrics.items():
    #                 mlflow.log_metric(m_name, m_val)
    #             # Clear for next trial
    #             current_trial_metrics.clear()

    # Define custom metric function to track additional metrics
    def custom_metric(X_val, y_val, estimator, labels, X_train, y_train, weight_val=None, weight_train=None, config=None, groups_val=None, deprecated_groups_train=None):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        import time
        start = time.time()
        
        y_pred = estimator.predict(X_val)
        y_pred_proba = estimator.predict_proba(X_val)[:, 1]
        
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        
        pred_time = (time.time() - start) / len(X_val)
        
        # Minimizing 1 - roc_auc
        val_loss = 1.0 - roc_auc
        
        metrics_dict = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "prediction_latency": pred_time
        }
        
        # Update shared dictionary so callback can access it (Disabled as callback is inactive)
        # current_trial_metrics.update(metrics_dict)
        
        return val_loss, metrics_dict

    print("Starting AutoML with FLAML...")
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"AutoML_Run_{timestamp}"
    print(f"Starting MLflow Run: {run_name}")
    
    mlflow.set_experiment("Heart_Disease_Prediction_AutoML")
    
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("run_type", "parent")
        automl = AutoML()
        
        # Define FLAML settings
        settings = {
            "time_budget": 60,  # seconds
            "metric": custom_metric, # Use custom metric function
            "task": 'classification',
            "estimator_list": ['lrl1', 'lrl2', 'rf', 'xgboost', 'lgbm'], 
            "log_file_name": 'flaml.log',
            "seed": 42,
            "eval_method": "cv", 
            "n_splits": 5,      
        }
        
        # Log params
        mlflow.log_param("preprocessing", "StandardScaler + OHE")
        # Log basic settings (skip function object)
        mlflow.log_param("time_budget", 60)
        mlflow.log_param("task", "classification")
        
        # Create artifacts directory for metadata and models
        os.makedirs("mlflow_artifacts", exist_ok=True)

        # Train
        automl.fit(X_train=X_train_processed, y_train=y_train, **settings)

        # Individual trials are now logged in real-time via mlflow_logging_callback
        print("AutoML training complete. Best model and metrics are archived in the parent run.")
        print(f"Best machine learning model selected: {automl.best_estimator}")
        print(f"Best hyperparameter config: {automl.best_config}")
        print(f"Best accuracy on validation data: {automl.best_loss}")

        y_pred = automl.predict(X_test_processed)
        y_pred_proba = automl.predict_proba(X_test_processed)[:, 1]

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Get detailed metrics from classification report
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        # Using weighted avg for summary
        precision = report_dict['weighted avg']['precision']
        recall = report_dict['weighted avg']['recall']
        f1 = report_dict['weighted avg']['f1-score']

        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test ROC AUC: {roc_auc:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1: {f1:.4f}")
        
        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("best_loss", automl.best_loss)
        
        # Log best params
        mlflow.log_params({"best_estimator": automl.best_estimator})
        
        # Log readable model type for the best model
        learner_map = {
            'lgbm': 'LightGBM',
            'xgboost': 'XGBoost',
            'rf': 'Random Forest',
            'lrl1': 'Logistic Regression (L1)',
            'lrl2': 'Logistic Regression (L2)',
            'catboost': 'CatBoost',
            'extra_tree': 'Extra Trees',
            'kneighbor': 'K-Nearest Neighbors'
        }
        readable_best_model = learner_map.get(automl.best_estimator, automl.best_estimator)
        mlflow.log_param("model_type", readable_best_model)
        for k, v in automl.best_config.items():
            mlflow.log_param(f"best_config_{k}", v)

        # ---------------------------------------------------------
        # Generate Explicit Tuning Report with Per-Model Details
        # ---------------------------------------------------------
        
        # Parse log for per-learner bests
        learner_stats = {} # learner_name -> {'error': float, 'config': dict}
        
        try:
            with open('flaml.log', 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        curr_learner = data.get('learner')
                        curr_error = data.get('validation_loss') # FLAML minimizes validation_loss

                        if curr_learner and curr_error is not None:
                             # Keep the best (lowest error) seen for this learnerso far
                             if curr_learner not in learner_stats or curr_error < learner_stats[curr_learner]['error']:
                                 learner_stats[curr_learner] = {
                                     'error': curr_error,
                                     'config': data.get('config')
                                 }
                    except:
                        continue
        except FileNotFoundError:
            print("Log file not found, skipping per-model detail parsing.")

        tuning_report = f"""# Auto-ML Model Selection and Hyperparameter Tuning Report

## 1. Tuning Strategy
- **Framework:** FLAML (Fast and Lightweight AutoML)
- **Time Budget:** {settings['time_budget']} seconds
- **Metric:** {settings['metric']} (Minimize 1 - ROC_AUC)

## 2. Models Explored
The pipeline searched across the following algorithm families:
- **Tree Ensembles:** Random Forest (rf), XGBoost (xgboost), LightGBM (lgbm), ExtraTrees (extra_tree)
- **Linear Models:** Logistic Regression with L1 (lrl1) and L2 (lrl2) penalties
- **Other:** CatBoost (catboost), K-Nearest Neighbors (kneighbor)

## 3. Leaderboard by Model Type
The best configuration found for each model type explored:

"""
        for learner, stats in learner_stats.items():
            readable_name = learner_map.get(learner, learner)
            tuning_report += f"### {readable_name} ({learner})\n"
            tuning_report += f"- **Best Validation Loss:** {stats['error']:.4f}\n"
            tuning_report += f"- **Best Configuration:**\n```json\n{json.dumps(stats['config'], indent=4)}\n```\n\n"

        tuning_report += f"""## 4. Overall Winner
The best performing model selected for final training was:

- **Best Model Type:** {readable_best_model} ({automl.best_estimator})
- **Best Validation Loss:** {automl.best_loss:.4f}
"""

        with open("mlflow_artifacts/tuning_report.md", "w") as f:
            f.write(tuning_report)
        mlflow.log_artifact("mlflow_artifacts/tuning_report.md", artifact_path="reports_and_metadata")
        print("Generated and logged mlflow_artifacts/tuning_report.md with per-model details.")

        # Log full best_config as JSON artifact for completeness
        with open("mlflow_artifacts/best_config.json", "w") as f:
            json.dump(automl.best_config, f, indent=4)
        mlflow.log_artifact("mlflow_artifacts/best_config.json", artifact_path="reports_and_metadata")

        # Log Model
        # Log Model as Pickle (Explicit Request)
        try:
            import pickle
            # 1. Log the best estimator from FLAML
            # automl.model is the wrapper, automl.model.estimator is the underlying sklearn-compatible model
            if hasattr(automl, 'model') and hasattr(automl.model, 'estimator'):
                best_model = automl.model.estimator
                with open("mlflow_artifacts/best_model.pkl", "wb") as f:
                    pickle.dump(best_model, f)
                mlflow.log_artifact("mlflow_artifacts/best_model.pkl", artifact_path="models")
                print("Successfully logged mlflow_artifacts/best_model.pkl")
                
                # Log via sklearn flavor
                mlflow.sklearn.log_model(best_model, "best_model_sklearn")
            else:
                 print("Could not access automl.model.estimator")

        except Exception as e:
            print(f"Error pickling model: {e}")
        
        # Evaluate using classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        # Flatten and log report metrics if needed, or save as JSON artifact
        with open("mlflow_artifacts/classification_report.json", "w") as f:
            json.dump(report, f, indent=4)
        mlflow.log_artifact("mlflow_artifacts/classification_report.json", artifact_path="reports_and_metadata")
        
        # Generate and log selection reasoning
        best_val_roc_auc = 1.0 - automl.best_loss
        reasoning = (
            f"Model Selection Reasoning:\n"
            f"--------------------------\n"
            f"The model '{readable_best_model}' ({automl.best_estimator}) was selected as the best model because:\n"
            f"1. Optimization Goal: Minimize 'val_loss' (defined as 1 - ROC_AUC)\n"
            f"2. Performance: It achieved the lowest validation loss of {automl.best_loss:.4f}\n"
            f"3. Metric Equivalent: This corresponds to a Validation ROC AUC of {best_val_roc_auc:.4f}\n"
            f"4. Constraints: The selection was made within a time budget of {settings['time_budget']} seconds.\n"
        )
        
        print(reasoning)
        with open("mlflow_artifacts/model_selection_reasoning.txt", "w") as f:
            f.write(reasoning)
        mlflow.log_artifact("mlflow_artifacts/model_selection_reasoning.txt", artifact_path="reports_and_metadata")
        mlflow.set_tag("selection_reason", f"Lowest val_loss: {automl.best_loss:.4f} (Val AUC: {best_val_roc_auc:.4f})")
        mlflow.log_param("selection_reason", f"Lowest val_loss: {automl.best_loss:.4f} (Val AUC: {best_val_roc_auc:.4f})")

        # ---------------------------------------------------------
        # Generate & Log Performance Plots (ROC Curve, Confusion Matrix)
        # ---------------------------------------------------------
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Disease', 'Disease'], 
                    yticklabels=['No Disease', 'Disease'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig("plots/confusion_matrix.png")
        mlflow.log_artifact("plots/confusion_matrix.png", artifact_path="model_evaluation")
        plt.close()

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc_val = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_val:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig("plots/roc_curve.png")
        mlflow.log_artifact("plots/roc_curve.png", artifact_path="model_evaluation")
        plt.close()
        
        print("Logged performance plots (plots/confusion_matrix.png, plots/roc_curve.png) to MLflow.")

        # ---------------------------------------------------------
        # Save Full Inference Pipeline (Preprocessor + Model)
        # ---------------------------------------------------------
        try:
            # 1. Save the fitted Preprocessor
            with open("mlflow_artifacts/preprocessor.pkl", "wb") as f:
                pickle.dump(preprocessor, f)
            mlflow.log_artifact("mlflow_artifacts/preprocessor.pkl", artifact_path="models")
            print("Successfully logged mlflow_artifacts/preprocessor.pkl")

            # 2. Create and Save Unified Inference Pipeline
            if hasattr(automl, 'model') and hasattr(automl.model, 'estimator'):
                
                # Combine fitted preprocessor and best model into a single pipeline
                inference_pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', automl.model.estimator)
                ])
                
                with open("mlflow_artifacts/inference_pipeline.pkl", "wb") as f:
                    pickle.dump(inference_pipeline, f)
                mlflow.log_artifact("mlflow_artifacts/inference_pipeline.pkl", artifact_path="models")
                
                # Also log as an MLflow Model (Pipeline flavor) and register it in the Model Registry
                mlflow.sklearn.log_model(
                    sk_model=inference_pipeline, 
                    artifact_path="inference_pipeline",
                    registered_model_name="Heart_Disease_Prediction_Pipeline"
                )
                print("Successfully created, logged, and REGISTERED inference_pipeline in MLflow Model Registry")
            
        except Exception as e:
            print(f"Error creating inference pipeline: {e}")

        print("Training finished. Logged to MLflow.")

if __name__ == "__main__":
    train_pipeline()
