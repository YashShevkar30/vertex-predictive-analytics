from pathlib import Path

class VertexConfig:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_RAW = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
    DATA_SCORED = PROJECT_ROOT / "data" / "scored"
    MODEL_DIR = PROJECT_ROOT / "models"
    REPORTS_DIR = PROJECT_ROOT / "reports"

    # Spark
    SPARK_APP_NAME = "VertexPredictiveAnalytics"
    SPARK_PARTITIONS = 8
    DAILY_PAYLOAD_MB = 18  # simulated

    # Clustering
    N_CLUSTERS = 5
    CLUSTER_FEATURES = ["total_events", "avg_session_duration",
                        "purchase_rate", "days_active", "avg_pages_per_session"]

    # Model
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    TARGET_COL = "churned"

config = VertexConfig()
