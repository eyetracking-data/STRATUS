# Local reference data (not redistributed)

The measured runtime inputs are derived from third-party ETDD70 and Autism eye-tracking data. This public repository intentionally does **not** distribute their coordinate-level reference segments; refer to `../../data/README.md` in the main STRATUS project for dataset acquisition details.

To repeat the benchmark, reconstruct or supply `clean_reference_segments.csv` from the licensed originals and pass `--reference /path/to/clean_reference_segments.csv`. The runtime runner requires the **exact** reference-file SHA-256:

`c4e4b81c948e28c80e0ba50c84e118e54081d5257cb3c5b7a2a9d1e85e467525`

The published measurements can be independently checked **without** this private input file:

```bash
python runtime_benchmark/verify_runtime_results.py
```

Extracted reference: 47 segments (ETDD70: 10; Autism: 37) each approximately 8 seconds, with the fixed participant split already committed in `results/tables/participant_split.csv`.

The data are not interchangeable with arbitrary CSVs: a nonmatching reference file will fail the SHA-256 check to prevent accidental claims of exact numerical reproduction.
