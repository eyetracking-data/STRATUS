# Local reference-data requirement

The GitHub artifact does not redistribute third-party raw eye-tracking data or
the coordinate-level `clean_reference_segments.csv` file.

Expected local v7 reference properties:

- columns: `dataset`, `source_file`, `participant_id`, `segment_id`, `time_s`,
  `x`, `y`, `case_segment`, `case_index`;
- 47 clean participant-specific segments;
- 10 ETDD70 and 37 Autism segments;
- approximately 8 seconds per segment;
- SHA-256:
  `c4e4b81c948e28c80e0ba50c84e118e54081d5257cb3c5b7a2a9d1e85e467525`.

The participant split and extraction summary are committed under
`results/tables/`. Use the existing dataset acquisition and extraction workflow
to recreate the file, then verify it with:

```bash
python scripts/verify_hybrid_results.py --reference /path/to/clean_reference_segments.csv
```
