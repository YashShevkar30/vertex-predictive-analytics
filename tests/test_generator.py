import pytest
import json
from vertex.data.generator import generate_behavioral_data
from vertex.config import config

def test_generation(tmp_path):
    config.DATA_RAW = tmp_path
    generate_behavioral_data(n_events=100, n_users=10, n_days=2)
    partitions = list(tmp_path.glob("date=*"))
    assert len(partitions) == 2
    jsonl = list(partitions[0].glob("*.jsonl"))
    assert len(jsonl) == 1
    with open(jsonl[0]) as f:
        event = json.loads(f.readline())
    assert "event_id" in event
    assert "user_id" in event
