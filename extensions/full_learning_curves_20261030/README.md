# Complete five-method learning curves

This extension adds GeoPool and GeoDirect at the 10%, 25%, and 50% budget settings for ten paired streams: 60 ensembles / 180 members. Combined with the original 140 ensembles, it provides 20 cells, 200 ensembles, and 600 members.

Use `results/summary_20_cells.csv` and `results/per_flow_200.csv` for the complete results. `results/descriptive_contrasts_by_ratio.csv` reports paired improvements. `results/validation_complete.json` records the original validation of all 60 new ensembles, and `results/aggregation_validation.json` records the original combined aggregation. These historical records describe the complete local experiment; they do not imply that its raw data or checkpoints are published here.

From the repository root, run `python scripts/verify_public_results.py` to check the public tables and `python scripts/plot_learning_curves.py` for the current figure. See [the reproduction guide](../../docs/REPRODUCING.md) for the external-input layout, training entrypoints, and provenance. The archived `scripts/prepare.py` is retained for provenance, not required to use the published task manifest.
