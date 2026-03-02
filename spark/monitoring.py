"""
Phase 3 — Drift Monitoring
============================
Tracks how model predictions and data statistics evolve over
time using Spark Structured Streaming windowed aggregations.

What it does
------------
  1. Collects each micro-batch to the driver via foreachBatch.
  2. Applies the fraud model on every event (plain pandas — no Arrow UDFs).
  3. Aggregates per-window statistics in tumbling time windows:
       - total events
       - predicted fraud count & rate
       - mean fraud probability
       - mean Amount
  4. Compares each window to a *baseline* fraud rate from training.
  5. Raises a DRIFT ALERT when the observed fraud rate deviates
     from baseline by more than `drift_threshold` (default: 3x).

Key ideas
---------
  - Drift happens gradually; monitoring is essential in production ML.
  - Models degrade silently — alerting on output distributions is a
    lightweight first line of defence.

Run with:
  spark-submit \\
      --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.8 \\
      spark/monitoring.py
"""

import os
import sys
import datetime
import joblib
import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    from_json,
)
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
KAFKA_BROKER    = os.getenv("KAFKA_BROKER",    "localhost:9092")
KAFKA_TOPIC     = os.getenv("KAFKA_TOPIC",     "transactions")
MODEL_PATH      = os.path.join(os.path.dirname(__file__), "..", "ml", "fraud_model.joblib")
STATS_PATH      = os.path.join(os.path.dirname(__file__), "..", "ml", "baseline_stats.joblib")

WINDOW_SECONDS  = int(os.getenv("WINDOW_SECONDS", "60"))

# Baseline fraud rate from the original dataset (~0.173 %)
DEFAULT_BASELINE_FRAUD_RATE = 0.00173
DRIFT_THRESHOLD_MULTIPLIER  = float(os.getenv("DRIFT_MULTIPLIER", "3.0"))

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]


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
# foreachBatch handler — prediction + windowed aggregation + drift
# ──────────────────────────────────────────────
_model = None


def make_foreach_batch(baseline_fraud_rate: float, drift_threshold: float, window_seconds: int):
    """Returns a foreachBatch function that scores, aggregates, and prints drift alerts."""

    def process_batch(batch_df, batch_id):
        global _model
        if batch_df.isEmpty():
            return

        # Load model once
        if _model is None:
            _model = joblib.load(MODEL_PATH)

        # Collect to driver
        pdf = batch_df.toPandas()

        # Predict
        X = pdf[FEATURE_COLS].values
        probs = _model.predict_proba(X)[:, 1]
        pdf["fraud_probability"] = probs.astype("float32")
        pdf["prediction"] = (probs > 0.5).astype(int)
        pdf["event_time"] = datetime.datetime.now()

        # Assign each event to a time window bucket
        pdf["window_bucket"] = pdf["event_time"].dt.floor(f"{window_seconds}s")

        # Aggregate per window
        agg = pdf.groupby("window_bucket").agg(
            total_events=("prediction", "count"),
            predicted_fraud_count=("prediction", "sum"),
            mean_fraud_probability=("fraud_probability", "mean"),
            mean_amount=("Amount", "mean"),
        )
        agg["predicted_fraud_rate"] = agg["predicted_fraud_count"] / agg["total_events"]

        print(f"\n{'='*70}")
        print(f"  MONITORING — Batch {batch_id}  |  Window results")
        print(f"{'='*70}")

        for window_start, row in agg.iterrows():
            window_end = window_start + pd.Timedelta(seconds=window_seconds)
            total      = int(row["total_events"])
            fraud_cnt  = int(row["predicted_fraud_count"])
            fraud_rate = row["predicted_fraud_rate"]
            mean_prob  = row["mean_fraud_probability"]
            mean_amt   = row["mean_amount"]

            ratio = fraud_rate / baseline_fraud_rate if baseline_fraud_rate > 0 else float("inf")
            drift_flag = ("DRIFT ALERT" if ratio > drift_threshold or ratio < (1 / drift_threshold)
                          else "OK")

            print(
                f"\n  Window : {window_start} -> {window_end}\n"
                f"  Events                : {total:>8,}\n"
                f"  Predicted frauds      : {fraud_cnt:>8,}\n"
                f"  Predicted fraud rate  : {fraud_rate:>10.4%}  "
                f"(baseline={baseline_fraud_rate:.4%},  ratio={ratio:.2f}x)\n"
                f"  Mean fraud probability: {mean_prob:>10.4f}\n"
                f"  Mean transaction amt  : {mean_amt:>10.2f}\n"
                f"  Drift status          : {drift_flag}"
            )

        print(f"\n{'='*70}\n")

    return process_batch


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}. Run ml/train_model.py first.")
        sys.exit(1)

    # Load baseline fraud rate from stats if available
    baseline_fraud_rate = DEFAULT_BASELINE_FRAUD_RATE
    if os.path.exists(STATS_PATH):
        stats = joblib.load(STATS_PATH)
        if "baseline_fraud_rate" in stats:
            baseline_fraud_rate = stats["baseline_fraud_rate"]

    # ── Spark session ──
    spark = (
        SparkSession.builder
        .appName("Phase3-DriftMonitoring")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print(f"[INFO] SparkSession created.")
    print(f"[INFO] Baseline fraud rate : {baseline_fraud_rate:.4%}")
    print(f"[INFO] Window duration     : {WINDOW_SECONDS}s")
    print(f"[INFO] Drift threshold     : {DRIFT_THRESHOLD_MULTIPLIER}x baseline")

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

    # ── Parse JSON ──
    parsed_df = (
        raw_df
        .select(from_json(col("value").cast(StringType()), schema).alias("d"))
        .select("d.*")
    )

    # ── foreachBatch: predict → aggregate → drift check ──
    process_batch = make_foreach_batch(
        baseline_fraud_rate=baseline_fraud_rate,
        drift_threshold=DRIFT_THRESHOLD_MULTIPLIER,
        window_seconds=WINDOW_SECONDS,
    )

    query = (
        parsed_df.writeStream
        .outputMode("append")
        .foreachBatch(process_batch)
        .trigger(processingTime="10 seconds")
        .start()
    )

    print(f"[INFO] Drift monitoring started. Listening on topic: {KAFKA_TOPIC}")
    query.awaitTermination()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()