# Generation benchmark

Measure generation and validation with a fresh output directory:

```bash
python benchmarks/generation.py \
  --output output/pilot \
  --rows 2000000 \
  --workers 2 \
  --batch-buildings 512 \
  --shard-buildings 4096 \
  --report output/pilot-benchmark.json
```

The report records the configuration and source fingerprint alongside:

| Field | Measurement |
| --- | --- |
| `decision_rows`, `buildings` | Generated decision and trajectory counts |
| `parquet_files`, `parquet_bytes` | File count and compressed bytes across both configurations |
| `bytes_per_decision` | Combined Parquet bytes divided by decision count |
| `generation_seconds` | Generation through finalization |
| `validation_seconds` | Full validation, including the saved report |
| `decisions_per_second` | Decision count divided by generation time |
| `parent_peak_rss_mib` | Parent-process lifetime peak memory, when available |
| `split_profiles` | Counts, action frequencies, missingness, outages and outcome summaries |

Run representative settings on the intended hardware and storage. Record worker
and batch settings with every measurement. Parent-process RSS measures the parent;
collect per-process or job-level memory when sizing a multi-worker deployment.
