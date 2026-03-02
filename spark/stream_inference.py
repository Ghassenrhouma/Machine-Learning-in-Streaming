"""
Phase 1 — Streaming Inference
==============================
Reads credit card transactions from Kafka in real time, applies a
pre-trained fraud-detection model to every incoming event, and
outputs predictions continuously to the console.

Key ideas
---------
  - One event → one prediction (stateless, event-level)
  - sklearn model applied inside foreachBatch (avoids Arrow UDF issues)
  - Low-latency micro-batch inference

Run with:
  spark-submit \\
      --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.8 \\
      spark/stream_inference.py
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
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_TOPIC  = os.getenv("KAFKA_TOPIC",  "transactions")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "fraud_model.joblib")

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]


# ──────────────────────────────────────────────
# Kafka JSON schema
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
# foreachBatch handler — runs on the driver
# ──────────────────────────────────────────────
_model = None  # loaded once lazily


def process_batch(batch_df, batch_id):
    global _model
    if batch_df.isEmpty():
        return

    # Load model once
    if _model is None:
        _model = joblib.load(MODEL_PATH)

    # Collect to driver as pandas DF
    pdf = batch_df.toPandas()

    X = pdf[FEATURE_COLS].values
    probs = _model.predict_proba(X)[:, 1]

    pdf["fraud_probability"] = probs.astype("float32")
    pdf["prediction"] = (probs > 0.5).astype(int)
    pdf["ingest_time"] = datetime.datetime.now()

    # Print key columns
    display_cols = ["ingest_time", "Time", "Amount", "fraud_probability", "prediction", "Class"]
    print(f"\n--- Batch {batch_id} ({len(pdf)} events) ---")
    print(pdf[display_cols].to_string(index=False, max_rows=20))
    print()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}. Run ml/train_model.py first.")
        sys.exit(1)

    # ── Spark session ──
    spark = (
        SparkSession.builder
        .appName("Phase1-StreamInference")
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

    print("[INFO] Streaming inference started. Waiting for events on topic:", KAFKA_TOPIC)
    query.awaitTermination()


if __name__ == "__main__":
    main()