"""
Behavioral Event Data Generator
==================================
Generates realistic semi-structured user behavioral data across
partitioned daily payloads simulating production clickstream ingestion.

Generates 125,000 rows across 30 daily partitions (~4,167 events/day).
"""
import os
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
from vertex.config import config

EVENT_TYPES = ["page_view", "click", "search", "add_to_cart",
               "purchase", "signup", "logout", "review"]
PAGES = ["/home", "/products", "/checkout", "/profile", "/search",
         "/deals", "/settings", "/support", "/blog", "/faq"]
DEVICES = ["mobile", "desktop", "tablet"]
CHANNELS = ["organic", "paid_search", "social", "email", "direct", "referral"]


def generate_behavioral_data(
    n_events: int = 125000,
    n_users: int = 5000,
    n_days: int = 30,
    seed: int = 42,
):
    """Generate partitioned daily behavioral event data."""
    rng = np.random.default_rng(seed)
    random.seed(seed)

    output_dir = config.DATA_RAW
    output_dir.mkdir(parents=True, exist_ok=True)

    base_date = datetime(2024, 6, 1)
    events_per_day = n_events // n_days

    # User profiles (some are churn-prone)
    user_churn_prob = {f"user_{i:05d}": rng.random() for i in range(n_users)}

    total_written = 0
    for day in range(n_days):
        current_date = base_date + timedelta(days=day)
        date_str = current_date.strftime("%Y-%m-%d")
        partition_dir = output_dir / f"date={date_str}"
        partition_dir.mkdir(parents=True, exist_ok=True)

        daily_events = []
        for _ in range(events_per_day):
            user_id = f"user_{rng.integers(0, n_users):05d}"
            churn_prob = user_churn_prob[user_id]

            # Churn-prone users have fewer events and lower engagement
            event_type = random.choice(EVENT_TYPES)
            if churn_prob > 0.7 and random.random() < 0.3:
                event_type = random.choice(["logout", "page_view"])

            event = {
                "event_id": f"evt_{total_written + len(daily_events):08d}",
                "user_id": user_id,
                "event_type": event_type,
                "page": random.choice(PAGES),
                "timestamp": (current_date + timedelta(
                    seconds=int(rng.integers(0, 86400))
                )).isoformat(),
                "device": random.choice(DEVICES),
                "channel": random.choice(CHANNELS),
                "session_duration_s": int(rng.exponential(180)),
                "pages_viewed": int(rng.poisson(5)),
                "revenue": round(float(rng.exponential(50)), 2)
                    if event_type == "purchase" else 0.0,
            }
            daily_events.append(event)

        # Write as JSON lines (partitioned daily ingestion)
        out_file = partition_dir / "events.jsonl"
        with open(out_file, "w") as f:
            for event in daily_events:
                f.write(json.dumps(event) + "\n")

        total_written += len(daily_events)

    # Generate user metadata with churn labels
    user_records = []
    for uid, cp in user_churn_prob.items():
        churned = 1 if cp > 0.65 else 0
        user_records.append({
            "user_id": uid,
            "signup_date": (base_date - timedelta(days=int(rng.integers(30, 365)))).isoformat(),
            "subscription_tier": random.choice(["free", "basic", "premium"]),
            "country": random.choice(["US", "UK", "DE", "FR", "IN", "JP", "BR"]),
            "churned": churned,
        })
    users_df = pd.DataFrame(user_records)
    users_df.to_csv(config.DATA_RAW / "users.csv", index=False)

    logger.info(f"Generated {total_written:,} events across {n_days} daily partitions")
    logger.info(f"Generated {len(user_records):,} user records "
               f"(churn rate: {users_df['churned'].mean():.1%})")


if __name__ == "__main__":
    generate_behavioral_data()
