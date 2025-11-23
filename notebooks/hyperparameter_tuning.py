"""
Hyperparameter Tuning for Best Models
Uses GridSearchCV to find optimal parameters
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score  # you can actually drop this import too

print("Loading data...")
train_df = pd.read_csv("data/train_pairs_rich.csv")

FEATURES = [
    "keyword_overlap",
    "skill_score",
    "experience_score",
    "education_score",
    "domain_match",
]

X_train = train_df[FEATURES]
y_train = train_df["label"]

# Custom scorer for ranking
# def ndcg_scorer(y_true, y_pred):
#     """Approximate NDCG for GridSearchCV"""
#     return roc_auc_score(y_true, y_pred)

# ============================================
# 1. RANDOM FOREST TUNING
# ============================================
print("\n" + "="*60)
print("Tuning Random Forest...")
print("="*60)

rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', None]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_params,
    cv=3,
    scoring="roc_auc",        # <--- changed
    verbose=2,
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

print(f"\nBest params: {rf_grid.best_params_}")
print(f"Best score: {rf_grid.best_score_:.4f}")

joblib.dump(rf_grid.best_estimator_, "models/model_rf_tuned.pkl")
print("Saved: ../models/model_rf_tuned.pkl")


# ============================================
# 2. GRADIENT BOOSTING TUNING
# ============================================
print("\n" + "="*60)
print("Tuning Gradient Boosting...")
print("="*60)

gb_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'subsample': [0.8, 1.0]
}

gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_params,
    cv=3,
    scoring="roc_auc",        # <--- changed
    verbose=2,
    n_jobs=-1
)

gb_grid.fit(X_train, y_train)

print(f"\nBest params: {gb_grid.best_params_}")
print(f"Best score: {gb_grid.best_score_:.4f}")

joblib.dump(gb_grid.best_estimator_, "models/model_gb_tuned.pkl")
print("Saved: ../models/model_gb_tuned.pkl")

# ============================================
# 3. SAVE TUNING RESULTS
# ============================================
results = pd.DataFrame({
    'Model': ['Random Forest (Tuned)', 'Gradient Boosting (Tuned)'],
    'Best Score': [rf_grid.best_score_, gb_grid.best_score_],
    'Best Params': [str(rf_grid.best_params_), str(gb_grid.best_params_)]
})

results.to_csv("data/hyperparameter_tuning_results.csv", index=False)
print("\n✅ Hyperparameter tuning complete!")
print(results.to_string(index=False))
