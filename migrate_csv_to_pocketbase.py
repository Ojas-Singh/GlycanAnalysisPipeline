#!/usr/bin/env python3
"""
One-time migration: Import GlycoShape_Inventory.csv into PocketBase glycan_submission collection.

Usage:
    python migrate_csv_to_pocketbase.py \
        --pocketbase-url http://localhost:8090 \
        --token <ADMIN_TOKEN> \
        --csv ../GlycoShape_Inventory.csv

Or via environment variables:
    POCKETBASE_URL=http://localhost:8090 \
    POCKETBASE_ADMIN_TOKEN=<token> \
    GLYCOSHAPE_INVENTORY_CSV=../GlycoShape_Inventory.csv \
    python migrate_csv_to_pocketbase.py
"""
import argparse
import csv
import sys
import time
from datetime import datetime

from lib.pocketbase import PocketBaseClient, COLLECTION


def parse_args():
    parser = argparse.ArgumentParser(description="Migrate GlycoShape_Inventory.csv to PocketBase")
    parser.add_argument("--pocketbase-url", default=None, help="PocketBase server URL")
    parser.add_argument("--token", default=None, help="PocketBase admin auth token")
    parser.add_argument("--csv", default=None, help="Path to GlycoShape_Inventory.csv")
    parser.add_argument("--dry-run", action="store_true", help="Print payloads without creating records")
    return parser.parse_args()


def parse_timestamp(ts_str):
    if not ts_str or not ts_str.strip():
        return ""
    for fmt in ("%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(ts_str.strip(), fmt)
            return dt.strftime("%Y-%m-%d %H:%M:%S.000Z")
        except ValueError:
            continue
    return ""


def parse_float(value):
    if not value or not value.strip():
        return None
    try:
        return float(value.strip())
    except ValueError:
        return None


def migrate_row(row):
    glycam_name = row.get("Full GLYCAM name of glycan being submitted.", "").strip()
    if not glycam_name:
        return {}

    glytoucan = row.get("What is the GlyTouCan ID of the glycan?", "").strip()
    if glytoucan in ("TODO", ""):
        glytoucan = ""

    payload = {
        "glycoshape_id": row.get("ID", "").strip(),
        "glycam_name": glycam_name,
        "source_type": "manual",
        "source_value": row.get("How will the data be transferred?", "").strip(),
        "status": "inbucket",
        "upload_status": "uploaded",
        "email": row.get("Email address", "").strip(),
        "simulation_length": parse_float(
            row.get("What is the aggregated length of the simulations?", "")
        ),
        "md_package": row.get("What MD package was used for the simulations?", "").strip(),
        "force_field": row.get("What force field was used for the simulations?", "").strip(),
        "temperature": parse_float(
            row.get("What temperature target was used for the simulations? ", "")
        ),
        "pressure": parse_float(
            row.get("What pressure target was used for the simulations?", "")
        ),
        "salt_concentration": parse_float(
            row.get("What NaCl concentration was used for the simulations?", "")
        ),
        "comments": row.get("Any comments that should be noted with the submission?", "").strip(),
        "glytoucan_id": glytoucan,
    }

    ts_raw = row.get("Timestamp", "").strip()
    if ts_raw:
        parsed_ts = parse_timestamp(ts_raw)
        if parsed_ts:
            payload["created"] = parsed_ts

    return payload


def main():
    import os

    args = parse_args()
    pb_url = args.pocketbase_url or os.environ.get("POCKETBASE_URL")
    token = args.token or os.environ.get("POCKETBASE_ADMIN_TOKEN")
    csv_path = args.csv or os.environ.get("GLYCOSHAPE_INVENTORY_CSV", "../GlycoShape_Inventory.csv")

    if not pb_url:
        print("ERROR: --pocketbase-url or POCKETBASE_URL required")
        sys.exit(1)
    if not token:
        print("ERROR: --token or POCKETBASE_ADMIN_TOKEN required")
        sys.exit(1)
    if not csv_path:
        print("ERROR: --csv or GLYCOSHAPE_INVENTORY_CSV required")
        sys.exit(1)

    client = PocketBaseClient(pb_url, token)

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} rows from {csv_path}")
    print(f"PocketBase: {pb_url}")
    print(f"Dry run: {args.dry_run}\n")

    success = 0
    skipped = 0
    failed = 0

    for i, row in enumerate(rows):
        gs_id = row.get("ID", "").strip()
        if not gs_id:
            print(f"  Row {i + 1}: skipped (no ID)")
            skipped += 1
            continue

        payload = migrate_row(row)
        if not payload:
            print(f"  {gs_id}: skipped (no glycam_name)")
            skipped += 1
            continue

        if args.dry_run:
            print(f"  {gs_id}: would create -> {payload.get('glycam_name')}")
            success += 1
            continue

        try:
            existing = client.get_record_by_glycoshape_id(gs_id)
            if existing:
                print(f"  {gs_id}: already exists (pb_id={existing['id']}), skipping")
                skipped += 1
                continue
        except Exception as exc:
            print(f"  {gs_id}: lookup failed ({exc}), skipping")
            skipped += 1
            continue

        try:
            result = client.request(
                "post",
                f"/api/collections/{COLLECTION}/records",
                json=payload,
            )
            print(f"  {gs_id}: created (pb_id={result['id']})")
            success += 1
        except Exception as exc:
            print(f"  {gs_id}: FAILED ({exc})")
            failed += 1

        if (i + 1) % 50 == 0:
            time.sleep(0.5)

    print(f"\nMigration complete: {success} created, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
