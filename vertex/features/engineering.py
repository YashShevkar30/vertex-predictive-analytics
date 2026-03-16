"""
Feature Engineering Pipeline
==============================
Transforms raw event data into user-level features for
predictive modeling. Includes temporal, behavioral, and
engagement features.
"""
import numpy as np
import pandas as pd
from loguru import logger
from vertex.config import config


def build_user_features(events_df: pd.DataFrame, users_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate event-level data into user-level features."""

    # Basic engagement
    user_agg = events_df.groupby("user_id").agg(
        total_events=("event_id", "count"),
        unique_pages=("page", "nunique"),
        unique_devices=("device", "nunique"),
        total_revenue=("revenue", "sum"),
        avg_session_duration=("session_duration_s", "mean"),
        max_session_duration=("session_duration_s", "max"),
        avg_pages_per_session=("pages_viewed", "mean"),
        days_active=("event_date", "nunique"),
    ).reset_index()

    # Event type ratios
    event_pivots = events_df.pivot_table(
        index="user_id", columns="event_type",
        values="event_id", aggfunc="count", fill_value=0
    )
    event_pivots.columns = [f"count_{c}" for c in event_pivots.columns]
    total = event_pivots.sum(axis=1)
    for col in event_pivots.columns:
        ratio_col = col.replace("count_", "ratio_")
        event_pivots[ratio_col] = event_pivots[col] / total

    # Purchase rate
    if "count_purchase" in event_pivots.columns:
        event_pivots["purchase_rate"] = event_pivots["count_purchase"] / total
    else:
        event_pivots["purchase_rate"] = 0.0

    # Channel distribution
    channel_pivots = events_df.pivot_table(
        index="user_id", columns="channel",
        values="event_id", aggfunc="count", fill_value=0
    )
    channel_pivots.columns = [f"channel_{c}" for c in channel_pivots.columns]

    # Temporal features
    time_features = events_df.groupby("user_id").agg(
        first_event=("event_timestamp", "min"),
        last_event=("event_timestamp", "max"),
        avg_hour=("hour", "mean"),
        std_hour=("hour", "std"),
    ).reset_index()
    time_features["tenure_days"] = (
        (time_features["last_event"] - time_features["first_event"]).dt.days
    )
    time_features["recency_days"] = (
        (pd.Timestamp.now() - time_features["last_event"]).dt.days
    )
    time_features.drop(columns=["first_event", "last_event"], inplace=True)
    time_features["std_hour"] = time_features["std_hour"].fillna(0)

    # Merge everything
    features = (
        user_agg
        .merge(event_pivots, on="user_id", how="left")
        .merge(channel_pivots, on="user_id", how="left")
        .merge(time_features, on="user_id", how="left")
        .merge(users_df[["user_id", "subscription_tier", "churned"]], on="user_id", how="left")
    )

    # Encode categoricals
    tier_map = {"free": 0, "basic": 1, "premium": 2}
    features["subscription_tier_encoded"] = features["subscription_tier"].map(tier_map).fillna(0)

    features = features.fillna(0)
    logger.info(f"Built {features.shape[1]} features for {len(features):,} users")
    return features


if __name__ == "__main__":
    events = pd.read_parquet(config.DATA_PROCESSED / "events")
    users = pd.read_csv(config.DATA_RAW / "users.csv")
    features = build_user_features(events, users)
    features.to_parquet(config.DATA_PROCESSED / "user_features.parquet", index=False)
