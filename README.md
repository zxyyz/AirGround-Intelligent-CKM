# 空地协同智能CKM
# Air–Ground Intelligent 6D Channel Knowledge Map

Paper: Measurement-Efficient 6D Channel Knowledge Map Construction via Air–Ground Cooperation (ICC 2027 manuscript).

本仓库包含论文的仿真源码、运行配置、汇总结果和结果图。正式实验采用已知建筑几何先验、几何路径级网络和空地协同测量。No raw CSI, per-query labels, trajectory caches, training records, or model checkpoints are uploaded.

## Main results / 主要结果

| Experiment | Learner / acquisition | Terminal GOSPA (mean ± SD, ten paired seeds) |
| --- | --- | --- |
| Baseline | Direct set decoder / independent coverage | 1.7978 ± 0.0228 |
| Geometry | Geometry-conditioned candidate-level learner / same independent sounding data | 0.8472 ± 0.0142 |
| Cooperative | Same geometry-conditioned learner / cooperative link acquisition | 0.7703 ± 0.0191 |

Relative final improvements averaged over paired seeds:
- Geometry versus Baseline: 52.87% [52.18%, 53.57%] paired 95% CI.
- Cooperative versus Geometry: 9.07% [7.71%, 10.42%] paired 95% CI.
- Ten out of ten seeds favor both improvements.

To verify published aggregate statistics without raw data, run:
    python tools/verify_published_results.py

## Directory layout

1. ckm_sim/ — geometry, propagation, MUSIC–LS, GOSPA, CKM learning, and active planning code.
2. experiments/paper_formal_240_20261017/scripts/ — paired 240-minute formal experiment code and tests.
3. experiments/paper_formal_240_20261017/configs/ — final experiment configuration, frozen MUSIC calibration, protocol, and scene manifest.
4. experiments/paper_formal_240_20261017/results/ — per-seed/per-round *aggregate* metrics, statistical comparisons, and budgets only.
5. experiments/paper_formal_240_20261017/figures/ — supplementary formal experiment figures.
6. experiments/gospa_scale_physical_resolution_20261018/ — metric sensitivity scripts, physical array-resolution calculation, and *aggregate* results.
7. tools/ — standard-library results verification.

## Dependencies and physics

Formal environment: Python 3.12.14, PyTorch 2.13.0, NumPy 2.5.4, SciPy 1.18.1, NVIDIA Sionna RT 2.2.0, Mitsuba 3.9.1, DrJit 1.5.0, h5py 3.16.0 and Matplotlib 3.11.2. See requirements.txt and the source dependencies.json.

Simulated 3.5-GHz channels include line-of-sight and up to two specular facade reflections. The learner receives public geometry but not material labels or hidden Sionna physical-ray identities. Sounding labels come from noisy MUSIC–LS. Noiseless MUSIC–LS provides the separate test references.

## Data and reproducibility boundary / 数据说明

Raw measurements, simulated per-query reference files, trajectories, caches, model checkpoints, trained weights, large scene meshes (PLY), and local logs are intentionally omitted.

The included Python experiment pipeline cannot execute a complete byte-for-byte rerun as-is: reconstruct the fixed glass/concrete scene meshes under experiments/paper_formal_240_20261017/assets/ and regenerate frozen query/reference caches and initialization states first. Scene parameters are preserved in configs/mesh_M2.json with portable relative asset paths; scene mesh files themselves are not included. No raw or trained data is included.

The saved GOSPA sensitivity summaries do not need raw prediction files to be read; rerunning all sensitivity scripts does require regenerating the frozen model predictions and references. The 15 tested evaluation scales retain the positive direction of both principal comparisons.

This repository currently contains research source and published summary results. No license is granted automatically.
