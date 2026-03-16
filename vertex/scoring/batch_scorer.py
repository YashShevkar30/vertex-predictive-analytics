"""
Batch Scoring Pipeline
========================
Loads trained model and scaler, scores all users, and generates
scored output with probabilities and risk tiers.
"""
import pandas as pd
import numpy as np
import joblib
from loguru import logger
from vertex.config import config
from vertex.models.train import get_feature_columns


def batch_score():
    model = joblib.load(config.MODEL_DIR / "best_model.pkl")
    scaler = joblib.load(config.MODEL_DIR / "scaler.pkl")

    feat_path = config.DATA_PROCESSED / "user_features_clustered.parquet"
    if not feat_path.exists():
        feat_path = config.DATA_PROCESSED / "user_features.parquet"
    df = pd.read_parquet(feat_path)

    feature_cols = get_feature_columns(df)
    X = scaler.transform(df[feature_cols].values)

    df["churn_prediction"] = model.predict(X)
    try:
        probs = model.predict_proba(X)[:, 1]
        df["churn_probability"] = probs
        df["risk_tier"] = pd.cut(
            probs,
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=["low", "medium", "high", "critical"],
        )
    except Exception:
        df["churn_probability"] = df["churn_prediction"].astype(float)
        df["risk_tier"] = "unknown"

    # Save scored output
    config.DATA_SCORED.mkdir(parents=True, exist_ok=True)
    output_path = config.DATA_SCORED / "scored_users.parquet"
    df.to_parquet(output_path, index=False)

    # Summary
    tier_summary = df["risk_tier"].value_counts()
    logger.info(f"\nScored {len(df):,} users")
    logger.info(f"\nRisk Tier Distribution:\n{tier_summary}")
    logger.info(f"\nOverall churn rate (predicted): {df['churn_prediction'].mean():.2%}")
    logger.info(f"Scored output saved to {output_path}")

    return df


if __name__ == "__main__":
    batch_score()
