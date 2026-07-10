from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
paths = {
    "ETDD70": Path(os.environ.get("STRATUS_ETDD70_DIR", ROOT / "data" / "raw" / "etdd70")),
    "Autism": Path(os.environ.get("STRATUS_AUTISM_DIR", ROOT / "data" / "raw" / "autism")),
}

failed = False
for name, path in paths.items():
    csv_count = len(list(path.glob("*.csv"))) if path.exists() else 0
    print(f"{name}: {path} ({csv_count} CSV files)")
    if csv_count == 0:
        failed = True

if failed:
    raise SystemExit("Dataset validation failed. See data/README.md.")
print("Dataset folders look usable.")
