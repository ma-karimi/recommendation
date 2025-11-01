from __future__ import annotations
from pathlib import Path
import datetime as dt
import os

import polars as pl

from dataframe_loader import load_products, load_order_items
from settings import load_config
from matomo_client import load_matomo_client


def run_pipeline(start_date: str | None = None, end_date: str | None = None) -> str:
    """
    Run the data pipeline: fetch Matomo aggregated data and write to Parquet.
    Returns the output path written.
    """
    cfg = load_config()
    client = load_matomo_client()

    sd = dt.date.fromisoformat(start_date) if start_date else dt.date.today() - dt.timedelta(days=7)
    ed = dt.date.fromisoformat(end_date) if end_date else dt.date.today()

    # for simplicity, fetch three datasets and persist separately
    events = client.fetch_events("range", f"{sd:%Y-%m-%d},{ed:%Y-%m-%d}")
    pageviews = client.fetch_pageviews("range", f"{sd:%Y-%m-%d},{ed:%Y-%m-%d}")
    goals = client.fetch_goals("range", f"{sd:%Y-%m-%d},{ed:%Y-%m-%d}")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)

    events_path = os.path.join(out_dir, f"matomo_events_{ts}.parquet")
    pageviews_path = os.path.join(out_dir, f"matomo_pageviews_{ts}.parquet")
    goals_path = os.path.join(out_dir, f"matomo_goals_{ts}.parquet")

    if events.height:
        events.write_parquet(events_path)
    else:
        events_path = ""

    if pageviews.height:
        pageviews.write_parquet(pageviews_path)
    else:
        pageviews_path = ""

    if goals.height:
        goals.write_parquet(goals_path)
    else:
        goals_path = ""

    return ",".join([p for p in [events_path, pageviews_path, goals_path] if p])


OUTPUT_DIR = Path(r"d:\PhpstormProjects\rochi-shop\storage\app\recommendation")


def save_products_parquet():
    df = load_products()
    if df.is_empty():
        print("[INFO] No products loaded from DB.")
        return
    out = OUTPUT_DIR / "products.parquet"
    df.write_parquet(out)
    print(f"[OK] Saved products to {out}")


def save_order_items_parquet(start_date: dt.date, end_date: dt.date):
    df = load_order_items(start_date, end_date)
    if df.is_empty():
        print("[INFO] No order items loaded from DB for given range.")
        return
    out = OUTPUT_DIR / f"order_items_{start_date}_{end_date}.parquet"
    df.write_parquet(out)
    print(f"[OK] Saved order items to {out}")


def main():
    cfg = load_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save DB-based datasets
    save_products_parquet()

    # Example: last 30 days order items
    today = dt.date.today()
    start = today - dt.timedelta(days=30)
    save_order_items_parquet(start, today)

    client = load_matomo_client()

    sd = dt.date.fromisoformat(start_date) if start_date else dt.date.today() - dt.timedelta(days=7)
    ed = dt.date.fromisoformat(end_date) if end_date else dt.date.today()

    # for simplicity, fetch three datasets and persist separately
    events = client.fetch_events("range", f"{sd:%Y-%m-%d},{ed:%Y-%m-%d}")
    pageviews = client.fetch_pageviews("range", f"{sd:%Y-%m-%d},{ed:%Y-%m-%d}")
    goals = client.fetch_goals("range", f"{sd:%Y-%m-%d},{ed:%Y-%m-%d}")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)

    events_path = os.path.join(out_dir, f"matomo_events_{ts}.parquet")
    pageviews_path = os.path.join(out_dir, f"matomo_pageviews_{ts}.parquet")
    goals_path = os.path.join(out_dir, f"matomo_goals_{ts}.parquet")

    if events.height:
        events.write_parquet(events_path)
    else:
        events_path = ""

    if pageviews.height:
        pageviews.write_parquet(pageviews_path)
    else:
        pageviews_path = ""

    if goals.height:
        goals.write_parquet(goals_path)
    else:
        goals_path = ""

    return ",".join([p for p in [events_path, pageviews_path, goals_path] if p])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run recommendation data pipeline (Matomo -> Parquet)")
    parser.add_argument("--start", dest="start_date", help="Start date (YYYY-MM-DD)", required=False)
    parser.add_argument("--end", dest="end_date", help="End date (YYYY-MM-DD)", required=False)
    args = parser.parse_args()

    output_paths = run_pipeline(args.start_date, args.end_date)
    print(output_paths)