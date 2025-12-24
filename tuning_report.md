# Auto-ML Model Selection and Hyperparameter Tuning Report

## 1. Tuning Strategy
- **Framework:** FLAML (Fast and Lightweight AutoML)
- **Time Budget:** 60 seconds
- **Metric:** <function train_pipeline.<locals>.custom_metric at 0x161f236a0> (Minimize 1 - ROC_AUC)

## 2. Models Explored
The pipeline searched across the following algorithm families:
- **Tree Ensembles:** Random Forest (rf), XGBoost (xgboost), LightGBM (lgbm), ExtraTrees (extra_tree)
- **Linear Models:** Logistic Regression with L1 (lrl1) and L2 (lrl2) penalties
- **Other:** CatBoost (catboost), K-Nearest Neighbors (kneighbor)

## 3. Leaderboard by Model Type
The best configuration found for each model type explored:

### Logistic Regression (L1) (lrl1)
- **Best Validation Loss:** 0.0929
- **Best Configuration:**
```json
{
    "C": 1.0
}
```

### Logistic Regression (L2) (lrl2)
- **Best Validation Loss:** 0.0899
- **Best Configuration:**
```json
{
    "C": 0.12500000000000006
}
```

### Random Forest (rf)
- **Best Validation Loss:** 0.0847
- **Best Configuration:**
```json
{
    "n_estimators": 17,
    "max_features": 0.22345754981405364,
    "max_leaves": 8,
    "criterion": "gini"
}
```

## 4. Overall Winner
The best performing model selected for final training was:

- **Best Model Type:** Random Forest (rf)
- **Best Validation Loss:** 0.0847
