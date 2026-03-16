import pytest
import pandas as pd
import numpy as np
from vertex.features.engineering import build_user_features

@pytest.fixture
def sample_data():
    events = pd.DataFrame({
        "event_id": [f"e{i}" for i in range(20)],
        "user_id": ["u1"]*10 + ["u2"]*10,
        "event_type": ["page_view", "click", "purchase", "search", "add_to_cart"]*4,
        "page": ["/home", "/products"]*10,
        "device": ["mobile", "desktop"]*10,
        "channel": ["organic", "paid_search"]*10,
        "session_duration_s": [120]*20,
        "pages_viewed": [5]*20,
        "revenue": [0]*18 + [50.0, 30.0],
        "event_timestamp": pd.date_range("2024-01-01", periods=20, freq="h"),
        "event_date": pd.date_range("2024-01-01", periods=20, freq="h").date,
        "hour": list(range(20)),
    })
    users = pd.DataFrame({
        "user_id": ["u1", "u2"],
        "subscription_tier": ["premium", "free"],
        "churned": [0, 1],
    })
    return events, users

def test_feature_count(sample_data):
    events, users = sample_data
    features = build_user_features(events, users)
    assert len(features) == 2
    assert "total_events" in features.columns
    assert "churned" in features.columns
