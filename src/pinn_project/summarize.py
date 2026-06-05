from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize experiment metrics.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="results/summary.csv")
    parser.add_argument("--exclude-prefix", action="append", default=[])
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.results_dir).glob("*/metrics.json")):
        if any(path.parent.name.startswith(prefix) for prefix in args.exclude_prefix):
            continue
        with path.open("r", encoding="utf-8") as f:
            rows.append(json.load(f))
    if not rows:
        raise SystemExit("No metrics.json files found.")

    keys = sorted({key for row in rows for key in row.keys()})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out} with {len(rows)} rows.")


if __name__ == "__main__":
    main()
