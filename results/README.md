# Simulation results

The current complete learning curves are in [`../extensions/full_learning_curves_20261030/results/`](../extensions/full_learning_curves_20261030/results/): `summary_20_cells.csv` contains five methods × four budgets, and `per_flow_200.csv` contains the 200 paired-stream measurements. The root `summary.csv` and `per_flow.csv` retain the original 14 cells / 140 ensembles for exact provenance checks; they are not the complete five-curve table.

## Main comparisons

`primary_contrasts.csv` and `paired_differences.csv` contain the four original full-budget paired tests, including Holm adjustment. Proposed improves over RawMap by 40.72% (95% CI [40.07%, 41.36%]) and over GeoDirect by 8.92% ([6.79%, 11.06%]). RawMap deteriorates relative to Coord by 0.59%, and GeoPool deteriorates relative to RawMap by 3.66%.

The extension's `descriptive_contrasts_by_ratio.csv` and `paired_differences_by_ratio.csv` report every budget. Proposed's paired improvement over GeoDirect is 15.72%, 12.36%, 9.43%, and 8.92% across increasing budgets. Ratios label frozen spatially nested subsets, so actual record fractions need not equal exactly 10/25/50%.

## Supplementary results

| File | Contents |
|---|---|
| `final/endpoint_summary.csv` | Full-budget GOSPA, recall, precision, missed/false paths, and conditional physical errors |
| `final/physical_error_summary.csv` | Per-stream physical errors, support counts, and quantiles |
| `final/gospa_components_per_flow.csv` | Additive gain/delay/direction/missed/false contributions to squared GOSPA |
| `scale_sensitivity_per_flow.csv` | All 15 normalization configurations with fixed predictions and recomputed associations |
| `scale_sensitivity_contrasts.csv` | Paired comparisons under every configuration |
| `power_recall.csv` | Recall stratified by reference path gain |
| `path_cap_coverage.csv`, `path_cap_frontend.csv` | Physical-ray coverage and frontend path-cap diagnostics |
| `training_fit_diagnostic.csv` | Training-fit diagnostics |
| `threshold_sensitivity.csv` | Fixed-threshold comparisons |
| `final_data_counts.csv`, `final_split_audit.json` | Record counts and spatial split checks |

Scale definitions are in `../configs/supplementary_protocol.json`. Proposed improves over RawMap by 23.908–52.847% and over GeoDirect by 4.998–18.432% across the 15 configurations. No models are retrained for these comparisons. GeoPool versus RawMap changes sign in two configurations.

At full budget, matched-path recall is 90.84% for Proposed and 87.97% for GeoDirect. Proposed's gain/delay/AoA/AoD RMSEs are 4.156 dB / 3.234 ns / 5.514° / 4.489°; GeoDirect's are 4.093 dB / 10.088 ns / 7.291° / 6.322°. These errors are conditional on optimal matching to the noiseless MUSIC–LS reference, and should be interpreted together with support and recall. Blank CSV values denote unavailable conditional metrics.

Coord and RawMap produce empty sets on 98.08% and 97.22% of test queries; the reference empty-set rate is 55.47%, and the always-empty RMS GOSPA is 1.741960. Proposed recalls 69.67% of reference paths in [−110, −105) dB. The three strongest physical rays above −110 dB retain 99.9146% of aggregate noncoherent power; 33 test queries contain more than three such rays.

The six GOSPA components sum to mean squared GOSPA within each stream. Averaging these components does not equal squaring the mean of per-stream RMS values. Statistics use ten paired noise/optimization streams in one city/material realization; test queries and individual paths are not independent experimental replicates. `Full` is the frozen key for Proposed throughout the CSVs.
