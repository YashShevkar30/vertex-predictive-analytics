"""
Model Training & Comparison
=============================
Trains and compares multiple classifiers for churn prediction:
- Naive Bayes (baseline)
- XGBoost
- LightGBM
- Gradient Boosting (sklearn)
- AdaBoost

Outputs a structured comparison table and saves the best model.
"""
import json
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import (
    classification_report, f1_score, precision_score,
    recall_score, roc_auc_score, accuracy_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from loguru import logger
from vertex.config import config


MODELS = {
    "naive_bayes": GaussianNB(),
    "xgboost": XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, use_label_encoder=False, eval_metric="logloss",
    ),
    "lightgbm": LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, verbose=-1,
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42,
    ),
    "adaboost": AdaBoostClassifier(
        n_estimators=100, learning_rate=0.1, random_state=42,
    ),
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Select numeric feature columns, excluding IDs and target."""
    exclude = {"user_id", "churned", "cluster", "subscription_tier",
               "event_date", "first_event", "last_event"}
    return [c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude]


def train_and_compare():
    # Load features
    feat_path = config.DATA_PROCESSED / "user_features_clustered.parquet"
    if not feat_path.exists():
        feat_path = config.DATA_PROCESSED / "user_features.parquet"
    df = pd.read_parquet(feat_path)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y = df[config.TARGET_COL].values

    logger.info(f"Features: {len(feature_cols)} | Samples: {len(X)} | "
               f"Positive rate: {y.mean():.2%}")

    # Scale features
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=y,
    )
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train and evaluate each model
    results = {}
    best_f1 = 0
    best_model_name = None
    best_model = None

    for name, model in MODELS.items():
        logger.info(f"Training {name}...")
        start = time.time()

        # Cross-validate
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_scaled, y_train,
                                   cv=cv, scoring="f1")

        # Full training
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - start

        # Predict
        y_pred = model.predict(X_test_scaled)
        y_prob = None
        try:
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        except Exception:
            pass

        # Metrics
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "f1": round(f1_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "cv_f1_mean": round(float(cv_scores.mean()), 4),
            "cv_f1_std": round(float(cv_scores.std()), 4),
            "training_time_s": round(train_time, 2),
        }
        if y_prob is not None:
            metrics["roc_auc"] = round(roc_auc_score(y_test, y_prob), 4)

        results[name] = metrics
        logger.info(f"  {name}: F1={metrics['f1']}, AUC={metrics.get('roc_auc', 'N/A')}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model_name = name
            best_model = model

    # Save best model
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, config.MODEL_DIR / "best_model.pkl")
    joblib.dump(scaler, config.MODEL_DIR / "scaler.pkl")

    # Save comparison
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.REPORTS_DIR / "model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print table
    logger.info(f"\n{'='*80}")
    logger.info("MODEL COMPARISON")
    logger.info(f"{'='*80}")
    comp_df = pd.DataFrame(results).T
    logger.info(f"\n{comp_df.to_string()}")
    logger.info(f"\nBest model: {best_model_name} (F1={best_f1})")

    return results, best_model_name


if __name__ == "__main__":
    train_and_compare()
