from vertex.config import config

def test_config():
    assert config.N_CLUSTERS == 5
    assert config.TEST_SIZE == 0.2
    assert config.TARGET_COL == "churned"
