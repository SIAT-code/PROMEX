#!/usr/bin/env python
"""Convert a simple PROMEX CSV file into the LMDB format used by property tasks."""

import argparse
import csv
import json
import shutil
from pathlib import Path

import lmdb


def parse_label(value):
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        return json.loads(value)
    if "," in value:
        parts = [x.strip() for x in value.split(",") if x.strip()]
        if parts and all(x.lstrip("-").isdigit() for x in parts):
            return [int(x) for x in parts]
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def main():
    parser = argparse.ArgumentParser(description="Convert CSV rows with name,seq,label columns to PROMEX LMDB.")
    parser.add_argument("--input", required=True, help="Input CSV file with columns: name, seq, label")
    parser.add_argument("--output", required=True, help="Output LMDB directory")
    parser.add_argument("--overwrite", action="store_true", help="Remove output directory if it already exists")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_path}. Use --overwrite to replace it.")
        shutil.rmtree(output_path)

    rows = []
    with input_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"name", "seq", "label"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        for row in reader:
            rows.append({"name": row["name"], "seq": row["seq"], "label": parse_label(row["label"])})

    env = lmdb.open(str(output_path), map_size=max(10485760, len(rows) * 65536))
    try:
        with env.begin(write=True) as txn:
            txn.put(b"length", str(len(rows)).encode())
            for index, row in enumerate(rows):
                txn.put(str(index).encode(), json.dumps(row).encode())
    finally:
        env.close()

    print(f"Wrote {len(rows)} records to {output_path}")


if __name__ == "__main__":
    main()
