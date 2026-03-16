"""
K-Means Behavioral Segmentation
==================================
Clusters users into behavioral segments based on engagement
features. Used for downstream feature enrichment and analysis.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from loguru import logger
from vertex.config import config
from vertex.features.engineering import build_user_features


def segment_users(features_df: pd.DataFrame = None) -> pd.DataFrame:
    if features_df is None:
        features_df = pd.read_parquet(config.DATA_PROCESSED / "user_features.parquet")

    cluster_cols = [c for c in config.CLUSTER_FEATURES if c in features_df.columns]
    if not cluster_cols:
        logger.warning("No clustering features found, using defaults")
        cluster_cols = ["total_events", "avg_session_duration", "days_active"]

    X = features_df[cluster_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=config.N_CLUSTERS,
        random_state=config.RANDOM_STATE,
        n_init=10,
    )
    labels = kmeans.fit_predict(X_scaled)
    features_df["cluster"] = labels

    # Cluster profiles
    profiles = features_df.groupby("cluster")[cluster_cols].mean()
    logger.info(f"\nCluster Profiles:\n{profiles.round(2)}")

    # Cluster sizes and churn rates
    cluster_stats = features_df.groupby("cluster").agg(
        size=("user_id", "count"),
        churn_rate=("churned", "mean"),
    )
    logger.info(f"\nCluster Stats:\n{cluster_stats.round(3)}")

    # Save
    features_df.to_parquet(config.DATA_PROCESSED / "user_features_clustered.parquet", index=False)
    logger.info(f"Segmented {len(features_df):,} users into {config.N_CLUSTERS} clusters")

    return features_df


if __name__ == "__main__":
    segment_users()
