"""
Phase 2 — Anomaly Detection on Streams
=========================================
Detects unusual credit card transactions in real time without relying
on labelled data.  Uses a statistical z-score approach based on
per-feature mean / std computed from the *normal* training examples.

Algorithm
---------
  For each incoming event:
    1. Compute z-score for Amount and every V-feature.
    2. Compute an aggregate anomaly_score = mean of absolute z-scores.
    3. Flag event as ANOMALY if anomaly_score > threshold  (default: 3.0).

Key ideas
---------
  - Anomalies detected at event level (stateless transformation)
  - No labels required — anomaly = deviation from normal behaviour
  - Streaming system reacts immediately to each event

Run with:
  spark-submit \\
      --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.8 \\
      spark/anomaly_detection.py
"""

import os
import sys
import datetime
import joblib
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
KAFKA_BROKER   = os.getenv("KAFKA_BROKER",   "localhost:9092")
KAFKA_TOPIC    = os.getenv("KAFKA_TOPIC",    "transactions")
STATS_PATH     = os.path.join(os.path.dirname(__file__), "..", "ml", "baseline_stats.joblib")
Z_THRESHOLD    = float(os.getenv("Z_THRESHOLD", "3.0"))   # flag as anomaly if score > this

FEATURE_COLS   = [f"V{i}" for i in range(1, 29)] + ["Amount"]


# ──────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────
def build_schema() -> StructType:
    fields = (
        [StructField("Time", DoubleType(), True)]
        + [StructField(f"V{i}", DoubleType(), True) for i in range(1, 29)]
        + [
            StructField("Amount", DoubleType(), True),
            StructField("Class",  DoubleType(), True),
        ]
    )
    return StructType(fields)


# ──────────────────────────────────────────────
# foreachBatch handler — z-score anomaly detection
# ──────────────────────────────────────────────
_stats = None  # loaded once lazily


def process_batch(batch_df, batch_id):
    global _stats
    if batch_df.isEmpty():
        return

    # Load baseline stats once
    if _stats is None:
        _stats = joblib.load(STATS_PATH)

    # Collect to driver as pandas DF
    pdf = batch_df.toPandas()

    X = pdf[FEATURE_COLS].values
    means = np.array([_stats["mean"][c] for c in FEATURE_COLS])
    stds  = np.array([_stats["std"][c]  for c in FEATURE_COLS])
    stds  = np.where(stds == 0, 1e-8, stds)

    z_scores = np.abs((X - means) / stds)
    pdf["anomaly_score"] = z_scores.mean(axis=1).astype("float32")
    pdf["is_anomaly"] = (pdf["anomaly_score"] > Z_THRESHOLD).astype(int)
    pdf["ingest_time"] = datetime.datetime.now()

    anomalies = pdf[pdf["is_anomaly"] == 1]

    display_cols = ["ingest_time", "Time", "Amount", "anomaly_score", "is_anomaly", "Class"]
    print(f"\n--- Batch {batch_id} ({len(pdf)} events, {len(anomalies)} anomalies) ---")
    if len(anomalies) > 0:
        print(anomalies[display_cols].to_string(index=False, max_rows=20))
    else:
        print("  (no anomalies detected)")
    print()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    if not os.path.exists(STATS_PATH):
        print(f"[ERROR] Baseline stats not found at {STATS_PATH}. Run ml/train_model.py first.")
        sys.exit(1)

    # ── Spark session ──
    spark = (
        SparkSession.builder
        .appName("Phase2-AnomalyDetection")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print("[INFO] SparkSession created.")

    schema = build_schema()

    # ── Read from Kafka ──
    raw_df = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BROKER)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    # ── Parse JSON payload ──
    parsed_df = (
        raw_df
        .select(from_json(col("value").cast(StringType()), schema).alias("d"))
        .select("d.*")
    )

    # ── Stream with foreachBatch (avoids Arrow UDF crashes) ──
    query = (
        parsed_df.writeStream
        .outputMode("append")
        .foreachBatch(process_batch)
        .trigger(processingTime="5 seconds")
        .start()
    )

    print(
        f"[INFO] Anomaly detection started (z-threshold={Z_THRESHOLD}). "
        f"Monitoring topic: {KAFKA_TOPIC}"
    )
    query.awaitTermination()


if __name__ == "__main__":
    main()