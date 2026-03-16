"""Spark Session Factory."""
def get_spark(app_name=None):
    try:
        from pyspark.sql import SparkSession
        from vertex.config import config
        return (SparkSession.builder
            .appName(app_name or config.SPARK_APP_NAME)
            .config("spark.sql.shuffle.partitions", str(config.SPARK_PARTITIONS))
            .config("spark.driver.memory", "2g")
            .master("local[*]")
            .getOrCreate())
    except ImportError:
        return None
