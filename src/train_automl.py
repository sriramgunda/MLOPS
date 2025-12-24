import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from flaml import AutoML
import mlflow
import mlflow.sklearn
import os
import json

def train_pipeline():
    # 1. Load Data
    print("Loading and preparing data...")
    try:
        from data_loader import load_data
    except ImportError:
        # Fallback for when running as a module or from different context
        import sys
        import os
        sys.path.append(os.path.join(os.getcwd(), 'src'))
        from data_loader import load_data

    # Generate/Load the dataset on the fly
    df = load_data()
    if df is None:
        print("Failed to load data.")
        return

    X = df.drop("target", axis=1)
    y = df["target"]

    # 2. Feature Engineering / Preprocessing
    # Defining feature groups
    categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

    print("Preprocessing data...")
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Fit and transform training data, transform test data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Convert back to dataframe for FLAML (optional but helpful for some estimators, though numpy is fine)
    # FLAML works fine with numpy arrays (CSR matrix from OHE).

    # Define custom callback to log each trial to MLflow
    def mlflow_logging_callback(val_loss, config, estimator, metric):
        # Start a nested run for each trial
        with mlflow.start_run(nested=True):
            # Log params
            mlflow.log_param("estimator", estimator)
            for k, v in config.items():
                mlflow.log_param(k, v)
            
            # Log metrics (val_loss is what FLAML minimizes, e.g., 1-AUC or error)
            mlflow.log_metric("val_loss", val_loss)
            if metric:
                 mlflow.log_metric("metric_value", metric)

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
            "val_accuracy": acc,
            "val_precision": prec,
            "val_recall": rec,
            "val_f1": f1,
            "val_roc_auc": roc_auc,
            "pred_time": pred_time
        }
        
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

        # Train
        automl.fit(X_train=X_train_processed, y_train=y_train, **settings)

        # Parse flaml.log to log each trial as a nested run
        print("Parsing FLAML log to track individual trials...")
        try:
            with open("flaml.log", "r") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        # Identify if it's a result line (usually has 'reward' or 'val_loss')
                        # FLAML log format varies, but normally has 'record_id', 'iter_no', 'logged_metric', etc.
                        # Standard FLAML log line example: {"record_id": 0, "iter_no": 0, "logged_metric": 0.5, "wall_clock_time": 0.1, "config": {...}, "learner": "xgboost", ...}
                        
                        if "config" in record and "learner" in record:
                            with mlflow.start_run(nested=True):
                                # Log learner and iteration
                                mlflow.log_param("learner", record.get("learner"))
                                
                                # Map learner to readable name
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
                                readable_name = learner_map.get(record.get("learner"), record.get("learner"))
                                mlflow.log_param("model_type", readable_name)
                                mlflow.log_param("iter_no", record.get("iter_no"))
                                
                                # Log config hyperparameters
                                config = record.get("config", {})
                                if config:
                                    for k, v in config.items():
                                        mlflow.log_param(k, v)
                                
                                # Log available metrics
                                # 'logged_metric' is usually the optimization metric (e.g. roc_auc or loss)
                                if "logged_metric" in record:
                                    # Note depending on min/max, this might be loss or score. 
                                    # FLAML minimizes 'val_loss' usually. 
                                    # 'logged_metric' corresponds to the `metric` defined in fit?
                                    metric_val = record["logged_metric"]
                                    if isinstance(metric_val, dict):
                                        # If it's a dict, log all keys
                                        for mk, mv in metric_val.items():
                                            try:
                                                mlflow.log_metric(f"logged_metric_{mk}", float(mv))
                                            except (ValueError, TypeError):
                                                pass
                                    else:
                                        try:
                                            mlflow.log_metric("logged_metric", float(metric_val))
                                        except (ValueError, TypeError):
                                            pass
                                
                                if "wall_clock_time" in record:
                                    mlflow.log_metric("wall_clock_time", record["wall_clock_time"])
                                    
                                if "validation_loss" in record:
                                     mlflow.log_metric("validation_loss", record["validation_loss"])

                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            print("flaml.log not found. Could not log individual trials.")
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

        with open("tuning_report.md", "w") as f:
            f.write(tuning_report)
        mlflow.log_artifact("tuning_report.md")
        print("Generated and logged tuning_report.md with per-model details.")

        # Log full best_config as JSON artifact for completeness
        with open("best_config.json", "w") as f:
            json.dump(automl.best_config, f, indent=4)
        mlflow.log_artifact("best_config.json")

        # Log Model
        # Log Model as Pickle (Explicit Request)
        try:
            import pickle
            # 1. Log the best estimator from FLAML
            # automl.model is the wrapper, automl.model.estimator is the underlying sklearn-compatible model
            if hasattr(automl, 'model') and hasattr(automl.model, 'estimator'):
                best_model = automl.model.estimator
                with open("best_model.pkl", "wb") as f:
                    pickle.dump(best_model, f)
                mlflow.log_artifact("best_model.pkl")
                print("Successfully logged best_model.pkl")
                
                # Log via sklearn flavor
                mlflow.sklearn.log_model(best_model, "best_model_sklearn")
            else:
                 print("Could not access automl.model.estimator")

        except Exception as e:
            print(f"Error pickling model: {e}")
        
        # Evaluate using classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        # Flatten and log report metrics if needed, or save as JSON artifact
        with open("classification_report.json", "w") as f:
            json.dump(report, f, indent=4)
        mlflow.log_artifact("classification_report.json")
        
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
        with open("model_selection_reasoning.txt", "w") as f:
            f.write(reasoning)
        mlflow.log_artifact("model_selection_reasoning.txt")
        mlflow.set_tag("selection_reason", f"Lowest val_loss: {automl.best_loss:.4f} (Val AUC: {best_val_roc_auc:.4f})")
        mlflow.log_param("selection_reason", f"Lowest val_loss: {automl.best_loss:.4f} (Val AUC: {best_val_roc_auc:.4f})")

        # ---------------------------------------------------------
        # Generate & Log Performance Plots (ROC Curve, Confusion Matrix)
        # ---------------------------------------------------------
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix, roc_curve, auc

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
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
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
        plt.savefig("roc_curve.png")
        mlflow.log_artifact("roc_curve.png")
        plt.close()
        
        print("Logged performance plots (confusion_matrix.png, roc_curve.png) to MLflow.")

        # ---------------------------------------------------------
        # Save Full Inference Pipeline (Preprocessor + Model)
        # ---------------------------------------------------------
        try:
            # 1. Save the fitted Preprocessor
            with open("preprocessor.pkl", "wb") as f:
                pickle.dump(preprocessor, f)
            mlflow.log_artifact("preprocessor.pkl")
            print("Successfully logged preprocessor.pkl")

            # 2. Create and Save Unified Inference Pipeline
            if hasattr(automl, 'model') and hasattr(automl.model, 'estimator'):
                from sklearn.pipeline import Pipeline
                
                # Combine fitted preprocessor and best model into a single pipeline
                inference_pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', automl.model.estimator)
                ])
                
                with open("inference_pipeline.pkl", "wb") as f:
                    pickle.dump(inference_pipeline, f)
                mlflow.log_artifact("inference_pipeline.pkl")
                
                # Also log as an MLflow Model (Pipeline flavor) which is very powerful
                mlflow.sklearn.log_model(inference_pipeline, "inference_pipeline")
                print("Successfully created and logged inference_pipeline.pkl (Preprocessor + Model)")
            
        except Exception as e:
            print(f"Error creating inference pipeline: {e}")

        print("Training finished. Logged to MLflow.")

if __name__ == "__main__":
    train_pipeline()
