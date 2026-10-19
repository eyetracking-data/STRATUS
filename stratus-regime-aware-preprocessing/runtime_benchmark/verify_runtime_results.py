"""Verify all published STRATUS runtime measurements without third-party input data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
REF_SHA = "c4e4b81c948e28c80e0ba50c84e118e54081d5257cb3c5b7a2a9d1e85e467525"
RUNNER_SHA = "414958e7966fdf206f97cb3a9222c354703d0c28a3ac4e3dd258692a2dc50f0c"

def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))

def close(actual, expected, tol=1e-8):
    if abs(actual - expected) > tol:
        raise AssertionError(f"Expected {expected}; found {actual}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results" / "runtime")
    ap.add_argument("--reference", type=Path, help="Optional private reference CSV for hash check")
    args = ap.parse_args()
    results = args.results
    rows = read_csv(results / "timings.csv")
    summary = read_csv(results / "summary.csv")
    training = read_csv(results / "training.csv")
    environment = json.loads((results / "environment.json").read_text(encoding="utf-8"))
    assert len(rows) == 4200, f"Expected 4200 timing measurements, got {len(rows)}"
    assert len(training) == 5
    assert environment["reference_sha256"] == REF_SHA
    assert environment["runner_sha256"] == RUNNER_SHA
    assert environment["training_cases"] == 240
    assert environment["training_shared_by_methods"] is True
    assert environment["smoke"] is False
    keys = {}
    for r in rows:
        assert r["dataset"] in ("ETDD70", "Autism")
        assert int(r["duration_setting_s"]) in (2, 4, 8)
        assert r["corruption"] in ("short_gap", "long_gap", "jitter", "spike", "mixed")
        assert 0 <= int(r["repeat"]) < 10
        assert r["method"] in ("Hybrid-P", "STRATUS-H")
        assert float(r["milliseconds"]) > 0
        k = (r["dataset"], r["participant"], r["case_segment"],
             r["duration_setting_s"], r["corruption"], r["repeat"])
        assert r["method"] not in keys.setdefault(k, {}), f"Duplicate key: {k}"
        keys[k][r["method"]] = float(r["milliseconds"])
    assert len(keys) == 2100
    assert all(set(v) == {"Hybrid-P", "STRATUS-H"} for v in keys.values())
    persons = {(r["dataset"], r["participant"]) for r in rows}
    assert len(persons) == 14
    assert sum(d == "ETDD70" for d, _ in persons) == 3
    assert sum(d == "Autism" for d, _ in persons) == 11
    for s in summary:
        chosen = [float(r["milliseconds"]) for r in rows
                  if (r["dataset"],r["duration_setting_s"],r["method"]) ==
                  (s["dataset"],s["duration_setting_s"],s["method"])]
        assert len(chosen) == int(s["count"])
        for col, val in (("median",median(chosen)), ("min",min(chosen)), ("max",max(chosen))):
            close(float(s[col]), val)
    assert len(summary) == 12
    for ds, hp, st, delta in (("ETDD70",22.07915,58.53875,36.20),
                              ("Autism",14.85710,22.30050,7.21)):
        subset = [r for r in rows if r["dataset"] == ds and r["duration_setting_s"] == "8"]
        close(median(float(r["milliseconds"]) for r in subset if r["method"] == "Hybrid-P"),hp)
        close(median(float(r["milliseconds"]) for r in subset if r["method"] == "STRATUS-H"),st)
        paired = [v["STRATUS-H"]-v["Hybrid-P"] for k,v in keys.items()
                  if k[0] == ds and k[3] == "8"]
        assert abs(median(paired)-delta) < .0051
    close(median(float(t["seconds"]) for t in training), 3.278017999982694,1e-6)
    if args.reference:
        actual = hashlib.sha256(args.reference.read_bytes()).hexdigest()
        assert actual == REF_SHA, f"Reference file hash differs: {actual}"
    source = HERE.parent / "scripts" / "run_hybrid_experiment.py"
    if source.is_file():
        assert hashlib.sha256(source.read_bytes()).hexdigest() == RUNNER_SHA, "Published source hash differs"
    print("PASS: 4,200 rows; 2,100 pairs; 12 aggregates; 14 held-out participants;")
    print("      two 8-second paired contrasts; 5 fits; reference/runner metadata.")
    if args.reference: print("PASS: local clean reference SHA-256 matches.")

if __name__ == "__main__":
    main()
