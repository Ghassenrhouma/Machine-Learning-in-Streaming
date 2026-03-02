"""
Kafka Producer — Credit Card Transactions
==========================================
Reads creditcard.csv row-by-row and publishes each transaction as a
JSON message to the Kafka topic `transactions`, simulating a real-time stream.

Usage:
  python kafka/producer.py [--delay 0.05] [--loop] [--topic transactions]

Arguments:
  --delay   Seconds to sleep between messages   (default: 0.05)
  --loop    Keep replaying the CSV indefinitely  (default: False)
  --topic   Kafka topic name                     (default: transactions)
  --broker  Kafka bootstrap server               (default: localhost:9092)
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def connect_producer(broker: str, retries: int = 10, wait: int = 5) -> KafkaProducer:
    """Attempt to connect to Kafka, retrying on failure."""
    for attempt in range(1, retries + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=[broker],
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks="all",
                retries=3,
            )
            print(f"[INFO] Connected to Kafka broker at {broker}")
            return producer
        except NoBrokersAvailable:
            print(f"[WARN] Broker not available (attempt {attempt}/{retries}). Retrying in {wait}s ...")
            time.sleep(wait)
    print("[ERROR] Could not connect to Kafka. Is the broker running?")
    sys.exit(1)


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[ERROR] Dataset not found at: {path}")
        print("Please download creditcard.csv from Kaggle and place it in data/")
        sys.exit(1)
    print(f"[INFO] Loading {path} ...")
    df = pd.read_csv(path)
    print(f"[INFO] {len(df):,} transactions loaded — fraud rate: {df['Class'].mean():.4%}")
    return df


def produce(df: pd.DataFrame, producer: KafkaProducer, topic: str, delay: float, loop: bool):
    """Iterate over the dataframe and publish each row to Kafka."""
    epoch = 0
    sent = 0

    while True:
        epoch += 1
        if loop:
            print(f"[INFO] Starting epoch {epoch} ...")

        for _, row in df.iterrows():
            message = row.to_dict()
            # Convert numpy types to native Python for JSON serialisation
            message = {k: float(v) if hasattr(v, "item") else v for k, v in message.items()}

            producer.send(topic, value=message)
            sent += 1

            if sent % 1000 == 0:
                print(f"[INFO] {sent:,} messages sent to topic '{topic}' ...")

            if delay > 0:
                time.sleep(delay)

        producer.flush()
        print(f"[INFO] Epoch {epoch} complete — total messages sent: {sent:,}")

        if not loop:
            break

    print(f"[INFO] Done. Total messages published: {sent:,}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kafka producer for credit card transactions.")
    parser.add_argument("--delay",  type=float, default=0.05,          help="Seconds between messages (default: 0.05)")
    parser.add_argument("--loop",   action="store_true",               help="Replay the CSV indefinitely")
    parser.add_argument("--topic",  type=str,   default="transactions", help="Kafka topic (default: transactions)")
    parser.add_argument("--broker", type=str,   default="localhost:9092", help="Kafka bootstrap server")
    return parser.parse_args()


def main():
    args = parse_args()

    df = load_csv(DATA_PATH)
    producer = connect_producer(broker=args.broker)

    print(f"[INFO] Publishing to topic '{args.topic}' with {args.delay}s delay (loop={args.loop}) ...")
    try:
        produce(df, producer, topic=args.topic, delay=args.delay, loop=args.loop)
    except KeyboardInterrupt:
        print("\n[INFO] Producer interrupted by user.")
    finally:
        producer.close()
        print("[INFO] Kafka producer closed.")


if __name__ == "__main__":
    main()