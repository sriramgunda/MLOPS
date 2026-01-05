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
import inspect

# Standardized Mapping for Learners
LEARNER_MAP = {
    'lgbm': 'LightGBM',
    'xgboost': 'XGBoost',
    'rf': 'Random Forest',
    'lrl1': 'Logistic Regression (L1)',
    'lrl2': 'Logistic Regression (L2)',
    #'catboost': 'CatBoost',
    #'extra_tree': 'Extra Trees',
    'kneighbor': 'K-Nearest Neighbors'
}

# Mapping for human-readable hyperparameter names in reports
HP_NAME_MAP = {
    'C': 'Regularization Constant (C)',
    'n_estimators': 'Number of Trees (n_estimators)',
    'max_depth': 'Maximum Depth (max_depth)',
    'learning_rate': 'Learning Rate (learning_rate)',
    'num_leaves': 'Number of Leaves (num_leaves)',
    'max_features': 'Max Features (max_features)',
    'min_child_samples': 'Min Child Samples (min_child_samples)',
    'reg_alpha': 'L1 Regularization (reg_alpha)',
    'reg_lambda': 'L2 Regularization (reg_lambda)',
}



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

    # Shared dictionary to pass metrics from custom_metric to the callback
    current_trial_metrics = {}



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
        
        
        # Minimizing 1 - (0.5 * Recall + 0.5 * ROC_AUC)
        # We prioritize Recall (Critical for medical diagnosis) while keeping ROC_AUC for stability
        # avoiding models that just predict class 1 everywhere (which would have perfect recall but poor AUC)
        composite_score = 0.5 * rec + 0.5 * roc_auc
        val_loss = 1.0 - composite_score
        
        metrics_dict = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "prediction_latency": pred_time
        }
        
        
        # Determine model type and log if MLflow run is active (FLAML manages the run when mlflow_logging=True)
        # Note: FLAML logs params automatically, but we want a standardized 'model_type'
        # try:
        #     if mlflow.active_run():
        #         est_name = estimator.__class__.__name__
        #         readable_name = est_name
        #         
        #         # Logic to map class name to readable name
        #         if 'LGBM' in est_name: readable_name = 'LightGBM'
        #         elif 'XGB' in est_name: readable_name = 'XGBoost'
        #         elif 'RandomForest' in est_name: readable_name = 'Random Forest'
        #         elif 'ExtraTrees' in est_name: readable_name = 'Extra Trees'
        #         elif 'KNeighbors' in est_name: readable_name = 'K-Nearest Neighbors'
        #         elif 'LogisticRegression' in est_name:
        #             penalty = getattr(estimator, 'penalty', None)
        #             if penalty == 'l1': readable_name = 'Logistic Regression (L1)'
        #             elif penalty == 'l2': readable_name = 'Logistic Regression (L2)'
        #             else: readable_name = 'Logistic Regression'
        #         
        #         # Log it
        #         # mlflow.log_param("model_type", readable_name)
        #         # Also ensure metrics are logged here if FLAML doesn't pick them up automatically from the return
        #         # for k, v in metrics_dict.items():
        #         #     mlflow.log_metric(k, v)
        #             
        # except Exception as e:
        #     # Swallow errors to avoid breaking training
        #     pass

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
            "metric": custom_metric, 
            "task": 'classification',
            "estimator_list": ['lrl1', 'lrl2', 'rf', 'xgboost', 'lgbm'], 
            "log_file_name": 'flaml.log',
            "seed": 42,
            "eval_method": "cv", 
            "n_splits": 5,
            "n_jobs": 1,
            "mlflow_logging": True, # Delegate run management to FLAML
            "mlflow_exp_name": "Heart_Disease_Prediction_AutoML",
        }
        # Log basic metadata
        mlflow.log_param("preprocessing", "StandardScaler + OHE")
        mlflow.log_param("time_budget", 60)
        mlflow.log_param("task", "classification")
        mlflow.log_param("metric_strategy", "Minimize 1 - (0.5 * Recall + 0.5 * ROC_AUC)")
        
        # Create artifacts directory
        artifact_dir = "best_model_artifacts"
        os.makedirs(artifact_dir, exist_ok=True)

        # Prepare fit arguments
        fit_kwargs = {
            "X_train": X_train_processed,
            "y_train": y_train,
            **settings
        }

        # Save AutoML configurations as artifact 
        config_artifact_path = os.path.join(artifact_dir, "automl_user_configurations.json")
        with open(config_artifact_path, "w") as f:
            # We filter out non-serializable objects (functions) if they exist
            serializable_settings = {k: (str(v) if callable(v) else v) for k, v in settings.items()}
            json.dump(serializable_settings, f, indent=4)
        mlflow.log_artifact(config_artifact_path, artifact_path=artifact_dir)

        print("Starting AutoML with FLAML (native MLflow logging enabled)...")

        # Train
        automl.fit(**fit_kwargs)

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
        readable_best_model = LEARNER_MAP.get(automl.best_estimator, automl.best_estimator)
        mlflow.log_param("model_type", readable_best_model)
        for k, v in automl.best_config.items():
            mlflow.log_param(f"best_config_{k}", v)

        # ---------------------------------------------------------
        # Generate Explicit Tuning Report with Per-Model Details
        # ---------------------------------------------------------
        
        # Parse log for per-learner bests and all trial history for visualization
        learner_stats = {} # learner_name -> {'error': float, 'config': dict}
        learner_stats = {} # learner_name -> {'error': float, 'config': dict}

        print("Parsing FLAML log for tuning report...")
        try:
            with open('flaml.log', 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        curr_learner = data.get('learner')
                        curr_error = data.get('validation_loss') # FLAML minimizes validation_loss
                        
                        if curr_learner and curr_error is not None:
                            # 1. Update Best Stats
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
- **Metric:** {settings['metric']} (Minimize 1 - (0.5 * Recall + 0.5 * ROC_AUC))

## 2. Models Explored
The pipeline searched across the following algorithm families:
- **Tree Ensembles:** Random Forest (rf), XGBoost (xgboost), LightGBM (lgbm), ExtraTrees (extra_tree)
- **Linear Models:** Logistic Regression with L1 (lrl1) and L2 (lrl2) penalties
- **Other:** CatBoost (catboost), K-Nearest Neighbors (kneighbor)

## 3. Leaderboard by Model Type
The best configuration found for each model type explored:

"""
        for learner, stats in learner_stats.items():
            readable_name = LEARNER_MAP.get(learner, learner)
            tuning_report += f"### {readable_name} ({learner})\n"
            tuning_report += f"- **Best Validation Loss:** {stats['error']:.4f}\n"
            
            # Format configuration with human-readable names
            readable_config = {}
            for k, v in stats['config'].items():
                name = HP_NAME_MAP.get(k, k)
                readable_config[name] = v
                
            tuning_report += f"- **Best Configuration:**\n```json\n{json.dumps(readable_config, indent=4)}\n```\n\n"

        tuning_report += f"""## 4. Overall Winner
The best performing model selected for final training was:

- **Best Model Type:** {readable_best_model} ({automl.best_estimator})
- **Best Validation Loss:** {automl.best_loss:.4f}
"""

        report_path = os.path.join(artifact_dir, "tuning_report.md")
        with open(report_path, "w") as f:
            f.write(tuning_report)
        mlflow.log_artifact(report_path, artifact_path=artifact_dir)
        print(f"Generated and logged {report_path}")

        # Log full best_config as JSON artifact
        config_path = os.path.join(artifact_dir, "best_config.json")
        with open(config_path, "w") as f:
            json.dump(automl.best_config, f, indent=4)
        mlflow.log_artifact(config_path, artifact_path=artifact_dir)

        # Log Model
        # Log Model as Pickle (Explicit Request)
        try:
            import pickle
            # 1. Log the best estimator from FLAML
            # automl.model is the wrapper, automl.model.estimator is the underlying sklearn-compatible model
            if hasattr(automl, 'model') and hasattr(automl.model, 'estimator'):
                best_model = automl.model.estimator
                pkl_path = os.path.join(artifact_dir, "best_model.pkl")
                with open(pkl_path, "wb") as f:
                    pickle.dump(best_model, f)
                mlflow.log_artifact(pkl_path, artifact_path=artifact_dir)
                print(f"Successfully logged {pkl_path}")
                
                # Log via sklearn flavor
                mlflow.sklearn.log_model(best_model, "best_model_sklearn")
            else:
                 print("Could not access automl.model.estimator")

        except Exception as e:
            print(f"Error pickling model: {e}")
        
        # Evaluate using classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_json_path = os.path.join(artifact_dir, "classification_report.json")
        with open(report_json_path, "w") as f:
            json.dump(report, f, indent=4)
        mlflow.log_artifact(report_json_path, artifact_path=artifact_dir)
        
        # Generate and log selection reasoning
        best_val_score = 1.0 - automl.best_loss
        reasoning = (
            f"Model Selection Reasoning:\n"
            f"--------------------------\n"
            f"The model '{readable_best_model}' ({automl.best_estimator}) was selected as the best model because:\n"
            f"1. Optimization Goal: Minimize 'val_loss' (defined as 1 - (0.5 * Recall + 0.5 * ROC_AUC))\n"
            f"   - This prioritizes models with high Recall (sensitivity) while maintaining good discrimination (AUC).\n"
            f"2. Validation Performance: It achieved the lowest validation loss of {automl.best_loss:.4f}\n"
            f"   - Validation Composite Score: {best_val_score:.4f}\n"
            f"3. Indicative Performance (Test Set):\n"
            f"   - Accuracy: {acc:.4f}\n"
            f"   - Recall: {recall:.4f}\n"
            f"   - ROC AUC: {roc_auc:.4f}\n"
            f"4. Constraints: The selection was made within a time budget of {settings['time_budget']} seconds.\n"
        )
        
        print(reasoning)
        reasoning_path = os.path.join(artifact_dir, "model_selection_reasoning.txt")
        with open(reasoning_path, "w") as f:
            f.write(reasoning)
        mlflow.log_artifact(reasoning_path, artifact_path=artifact_dir)
        # Log selection metrics as tags for easy retrieval
        mlflow.set_tag("selection_reason", f"Best Val Loss: {automl.best_loss:.4f}")
        mlflow.log_param("selection_reason", f"Score: {best_val_score:.4f} (Rec:{recall:.4f}, AUC:{roc_auc:.4f})")

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
        cm_path = os.path.join(artifact_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path, artifact_path=artifact_dir)
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
        roc_path = os.path.join(artifact_dir, "roc_curve.png")
        plt.savefig(roc_path)
        mlflow.log_artifact(roc_path, artifact_path=artifact_dir)
        plt.close()
        
        print(f"Logged performance plots to {artifact_dir} in MLflow.")

        # ---------------------------------------------------------
        # Save Full Inference Pipeline (Preprocessor + Model)
        # ---------------------------------------------------------
        try:
            # 1. Save the fitted Preprocessor
            prep_path = os.path.join(artifact_dir, "preprocessor.pkl")
            with open(prep_path, "wb") as f:
                pickle.dump(preprocessor, f)
            mlflow.log_artifact(prep_path, artifact_path=artifact_dir)
            print(f"Successfully logged {prep_path}")

            # 2. Create and Save Unified Inference Pipeline
            if hasattr(automl, 'model') and hasattr(automl.model, 'estimator'):
                
                # Combine fitted preprocessor and best model into a single pipeline
                inference_pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', automl.model.estimator)
                ])
                
                inf_path = os.path.join(artifact_dir, "inference_pipeline.pkl")
                with open(inf_path, "wb") as f:
                    pickle.dump(inference_pipeline, f)
                mlflow.log_artifact(inf_path, artifact_path=artifact_dir)
                
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
