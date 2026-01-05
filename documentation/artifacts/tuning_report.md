# Auto-ML Model Selection and Hyperparameter Tuning Report

## 1. Tuning Strategy
- **Framework:** FLAML (Fast and Lightweight AutoML)
- **Time Budget:** 60 seconds
- **Metric:** <function train_pipeline.<locals>.custom_metric at 0x7f6eabd4af80> (Minimize 1 - (0.5 * Recall + 0.5 * ROC_AUC))

## 2. Models Explored
The pipeline searched across the following algorithm families:
- **Tree Ensembles:** Random Forest (rf), XGBoost (xgboost), LightGBM (lgbm), ExtraTrees (extra_tree)
- **Linear Models:** Logistic Regression with L1 (lrl1) and L2 (lrl2) penalties
- **Other:** CatBoost (catboost), K-Nearest Neighbors (kneighbor)

## 3. Leaderboard by Model Type
The best configuration found for each model type explored:

### Logistic Regression (L1) (lrl1)
- **Best Validation Loss:** 0.1597
- **Best Configuration:**
```json
{
    "Regularization Constant (C)": 0.5266038477325524
}
```

### XGBoost (xgboost)
- **Best Validation Loss:** 0.1534
- **Best Configuration:**
```json
{
    "Number of Trees (n_estimators)": 4,
    "max_leaves": 44,
    "min_child_weight": 7.702975277431408,
    "Learning Rate (learning_rate)": 0.7135003922079828,
    "subsample": 0.7164280511605149,
    "colsample_bylevel": 0.8984480356505516,
    "colsample_bytree": 0.6993108042372532,
    "L1 Regularization (reg_alpha)": 0.006896809387654332,
    "L2 Regularization (reg_lambda)": 0.476223299246811
}
```

## 4. Overall Winner
The best performing model selected for final training was:

- **Best Model Type:** XGBoost (xgboost)
- **Best Validation Loss:** 0.1534
