"""
Spark Ingestion Pipeline
==========================
Reads partitioned daily JSON event data, validates schema,
applies type casting, and writes consolidated Parquet files
for downstream analytics.

Simulates processing 18GB daily payloads by reading partitioned
JSON lines files in a streaming-compatible pattern.
"""
import os
from loguru import logger
from vertex.config import config
from vertex.spark.session import get_spark


def create_event_schema():
    from pyspark.sql.types import (
        StructType, StructField, StringType, DoubleType,
        IntegerType, TimestampType
    )
    return StructType([
        StructField("event_id", StringType(), False),
        StructField("user_id", StringType(), False),
        StructField("event_type", StringType(), True),
        StructField("page", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("device", StringType(), True),
        StructField("channel", StringType(), True),
        StructField("session_duration_s", IntegerType(), True),
        StructField("pages_viewed", IntegerType(), True),
        StructField("revenue", DoubleType(), True),
    ])


def ingest_events():
    spark = get_spark("VertexIngestion")
    if not spark:
        logger.warning("PySpark not available, using pandas fallback")
        _pandas_fallback()
        return

    schema = create_event_schema()
    raw_path = str(config.DATA_RAW / "date=*" / "*.jsonl")

    logger.info(f"Reading partitioned events from {raw_path}")
    df = spark.read.schema(schema).json(raw_path)

    from pyspark.sql import functions as F

    # Type casting and validation
    df = (df
        .withColumn("event_timestamp", F.to_timestamp("timestamp"))
        .withColumn("event_date", F.to_date("event_timestamp"))
        .withColumn("hour", F.hour("event_timestamp"))
        .withColumn("day_of_week", F.dayofweek("event_timestamp"))
        .drop("timestamp")
    )

    # Quality check: count nulls
    null_counts = {c: df.filter(F.col(c).isNull()).count() for c in ["event_id", "user_id"]}
    logger.info(f"Null check: {null_counts}")

    # Write consolidated Parquet
    out_path = str(config.DATA_PROCESSED / "events")
    df.write.mode("overwrite").partitionBy("event_date").parquet(out_path)

    total = df.count()
    logger.info(f"Ingested {total:,} events to {out_path}")
    spark.stop()


def _pandas_fallback():
    """Fallback ingestion using pandas for environments without Spark."""
    import pandas as pd
    import json

    all_events = []
    raw_dir = config.DATA_RAW
    for partition in sorted(raw_dir.glob("date=*")):
        for f in partition.glob("*.jsonl"):
            with open(f) as fh:
                for line in fh:
                    all_events.append(json.loads(line))

    df = pd.DataFrame(all_events)
    df["event_timestamp"] = pd.to_datetime(df["timestamp"])
    df["event_date"] = df["event_timestamp"].dt.date
    df["hour"] = df["event_timestamp"].dt.hour
    df["day_of_week"] = df["event_timestamp"].dt.dayofweek
    df.drop(columns=["timestamp"], inplace=True)

    out = config.DATA_PROCESSED / "events"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "events.parquet", index=False)
    logger.info(f"Pandas fallback: ingested {len(df):,} events")


if __name__ == "__main__":
    ingest_events()
