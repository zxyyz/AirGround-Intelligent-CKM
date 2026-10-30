# Reproducing the reported results

## Available directly from this repository

```bash
python -m pip install numpy scipy matplotlib
python scripts/verify_public_results.py
python scripts/plot_learning_curves.py
```

The checker recomputes all 20 GOSPA means, sample SDs and 95% t intervals from 200 per-stream rows, checks paired improvements at every budget, verifies that the original 140 rows are unchanged, and checks all 15 scale comparisons. It also checks the frozen scientific-source hashes. No training or measurement data are needed.

The plotting command reads `extensions/full_learning_curves_20261030/results/summary_20_cells.csv` and writes `figures/five_method_learning_curves.pdf`, `.png`, and `plotted_data.csv`. Install Times New Roman through your operating system before plotting; the font is not redistributed. The script fails explicitly when the font is unavailable. This is the current five-curve manuscript figure (Figure 4); the simulation scene is Figure 3. Older plotting routines inside the frozen experiment source are retained for provenance, but are not the current manuscript plotting entrypoint.

## Executed experiment

- 60 kHz OFDM subcarrier grid, comb 8, 128 pilot frequencies, actual pilot spacing 480 kHz.
- Nominal sounding bandwidth 61.44 MHz; smoothing bandwidth 30.72 MHz.
- Reference SNR 30 dB at path gain −90 dB.
- 9,556 full-budget training records per stream at 7,181 distinct joint positions; 512 development and 4,096 shared test queries.
- One spatial split: three training groups, one development group, one test group. Repeated observations of a position remain in the same group.
- Four nested budgets, averaging 1,041.4 / 2,457.1 / 4,824.0 / 9,556 observations.
- Five methods × four budgets × ten paired streams = 200 formal ensembles; three members each; 56,800 updates per member, initialized from scratch.
- 64 geometric candidate slots plus one free proposal; at most three retained output paths; three-member medoid inference.
- GOSPA scales: 5 dB, 25 ns, 15° for each direction; cutoff 3.

`configs/freeze.json` records the final learning rates, thresholds, scientific-source hashes, input hashes, and batch hashes. `configs/protocol.json` preserves the initial development plan (including its initial unfrozen status and original three-curve scope); the final experiment comprises that original 140-ensemble plan plus the fixed-protocol 60-ensemble extension. Development used 45 additional ensembles. Extension comparisons at the added sparse budgets were computed after the original confirmation results were viewed, without retuning. They are descriptive; the original full-budget primary tests are unchanged.

## External inputs for full simulation replay

The following inputs are currently **not published**:

- `assets/`: frozen meshes and triangle metadata.
- `cache/`: frozen coordinates, noisy observations, reference path sets, geometry features, and physical-ray records.
- `manifests/batches/`: common minibatch and nested-subset arrays.
- `runs/` and `extensions/full_learning_curves_20261030/runs/`: checkpoints, training records, evaluations, and per-query predictions.

The repository therefore does not yet provide a standalone end-to-end reproduction download. Original orchestration also expects archived completion evidence such as `results/new_measurement_resources_0.json`; keep this evidence when preparing a replay copy. With a complete matching experiment archive available locally, use the layout above and verify its hashes against `configs/freeze.json`. Do not substitute unrelated historical data.

For reaggregation of an archived experiment, the relevant source entrypoints are:

```bash
python scripts/analyze.py
python scripts/validate_results.py
python scripts/revision_metrics.py
python extensions/full_learning_curves_20261030/scripts/aggregate.py
python scripts/plot_learning_curves.py
```

The first three operate on the original 140 ensembles. Extension aggregation additionally needs the archived extension validation output and its per-query CSVs; these per-query inputs are not in this checkout. `tools/hyperparameters.py` contains physical-path coverage, frontend-cap, and fixed-prediction scale-sensitivity routines.

For training with the complete frozen inputs, use a separate copy with empty run directories, create `logs/`, `results/`, and the extension's `logs/`, `cache/`, `results/`, and `audit/` directories, then run:

```bash
python scripts/revision_execute.py
python extensions/full_learning_curves_20261030/scripts/execute.py
```

These are the original two-GPU schedulers; the extension assigns five paired streams per GPU. The freeze file selects the final hyperparameters and training budget. Completed-run markers and checkpoints must not be copied into a fresh-training directory. Single-GPU scheduling requires adapting the scheduler and recording that change. Hardware/library differences may affect floating-point results. `revision_rebuild.py setup` is a historical initialization step that depends on the earlier experiment; it is unnecessary when the frozen coordinates and inputs are already supplied.

## Scene visualization

With the complete frozen experiment inputs and Mitsuba/Dr.Jit/Pillow installed:

```bash
python scripts/render_scene.py --experiment /path/to/experiment --output output/scene
```

The script reads archived query 28 and the actual frozen scene; it does not rerun ray tracing. It draws all 11 candidates retained for this query (4 geometrically valid, 7 invalid), together with the strongest archived direct, single-reflection, and double-reflection ray. The 64 candidate slots are an upper bound, not a requirement to produce 64 paths. Three valid candidates overlap the highlighted rays. Solid/dashed dark-gray lines distinguish valid/invalid candidates, and the obscuring building is translucent. Labels are TX UAV and RX UGV. `--reuse-render` reuses `scene_render.png` in the output directory.

## Source provenance

Scientific modules covered by `configs/freeze.json` retain their executed bytes. Publication changes add documentation, summary verification, and portable figure entrypoints. In the extension task manifest, machine-specific batch paths are now repository-relative; the protocol's baseline is also relative. `execution_code_manifest.json` records both the original hashes of these two metadata files and their public hashes. No model parameters, observations, or frozen metric settings were changed.
