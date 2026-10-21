# Method and implementation details for the geometry-conditioned CKM learner

This document supplies implementation details omitted from Section III of the paper. It describes the **M3 candidate-level learner used in `experiments/paper_formal_240_20261017`**, not the earlier attention/gated `CandidateSetCKM` or `HybridCKM` ablations. The documentation does not change any executable code, training objective, measured data, or reported result.

## 1. Source map

All paths below are relative to the repository root.

| Component | Source |
|---|---|
| Public direct/single/double-reflection candidates and 40-dimensional features | [`public_multibounce_geometry.py`](../experiments/paper_formal_240_20261017/scripts/public_multibounce_geometry.py) |
| Formal feature construction and model selection | [`formal_models.py`](../experiments/paper_formal_240_20261017/scripts/formal_models.py) and [`models.py`](../experiments/paper_formal_240_20261017/scripts/models.py) |
| Actual M3 network, matching loss, and training | [`candidate_level.py`](../experiments/paper_formal_240_20261017/scripts/candidate_level.py) |
| Replay weights | `replay_weights` in [`learning.py`](../experiments/paper_formal_240_20261017/scripts/learning.py) |
| Probability thresholding and duplicate suppression | [`decoding.py`](../experiments/paper_formal_240_20261017/scripts/decoding.py) |
| Set-wise committee aggregation | `committee_distances` in [`ckm_sim/model.py`](../ckm_sim/model.py) |
| Formal coverage-only acquisition | [`coverage.py`](../experiments/paper_formal_240_20261017/scripts/coverage.py) |
| Initialization, acquisition windows, and model-state continuation | [`runner.py`](../experiments/paper_formal_240_20261017/scripts/runner.py), [`ckm_sim/v14_planning.py`](../ckm_sim/v14_planning.py), and [`ckm_sim/planning.py`](../ckm_sim/planning.py) |

The source blobs checked when writing this note were `2fd4fde047fc85064072c7e72e35c30821283801` (`candidate_level.py`), `d9a6f1a68379931e1c8e6faa970121d466366eb8` (`public_multibounce_geometry.py`), `f6a612f4e67d6c9d1583386c4019848445af0b5f` (`decoding.py`), `5253d85d0a4911c76e04b5f0532ae7d528f0b6c7` (`learning.py`), and `8d0f848a9e9f587f363020ab2a7758145d6b9638` (`ckm_sim/model.py`).

## 2. Known geometry and path conventions

The public map contains building surfaces, roads, and permitted airspace. Learner inputs do not include material parameters, simulator path identities, or held-out reference channels. For a query, the six coordinates are the UAV transmitter and UGV receiver positions, not six independently unconstrained physical degrees of freedom: the UGV remains on legal roads at its antenna height.

A candidate has reflection order `J` equal to 0, 1, or 2. With vertices

$$s_{j,0}=p_U,\quad s_{j,1},\ldots,s_{j,J},\quad s_{j,J+1}=p_G,$$

its geometric delay is

$$\tau_j^{\mathrm{geo}}=\frac{1}{c_0}\sum_{h=0}^{J}\|s_{j,h+1}-s_{j,h}\|_2.$$

The raw departure direction is `(s[j,1] - p_U) / norm(s[j,1] - p_U)`. The raw arrival direction is `(s[j,J] - p_G) / norm(s[j,J] - p_G)`: it points **from the receiver back toward the incoming source**, opposite to the final segment's propagation direction. For a direct candidate, the other endpoint replaces the reflection point. The horizontal-array canonical representation sets the arrival z component nonnegative and the departure z component nonpositive.

The image-method implementation checks finite facade bounds, outward-side and specular-reflection conditions, clear physical segments, and (for double reflections) absence of a self-crossing path. Candidate records are ordered by validity first, then length and facade identities; at most 64 records are retained. Invalid and padded records are masked in the learner. A geometrically feasible candidate is only a hypothesis: its observability and gain are learned from the measured channel.

The geometric vertices are **not** updated by the neural model. Bounded delay/direction residuals change predicted channel features; the diagrams' corrected direction arrows must not be read as displaced reflection points or a repaired physical map.

## 3. Exact input layout

The input starts with the six endpoint coordinates normalized by `[240,240,120,240,240,120]`. It then contains `64 x 40` candidate features, flattened. Reflection points use the three-coordinate scale `[240,240,120]`. The following offsets are zero-based within each candidate:

| Offset(s) | Feature |
|---|---|
| 0 | Geometric validity |
| 1 | Reflection order divided by 2 |
| 2 | Geometric delay divided by `tau_max` |
| 3–5 | Canonical receive direction |
| 6–8 | Canonical transmit direction |
| 9 | Nonpadding indicator |
| 10–11 | Ordered facade IDs, encoded as `(id+1)/number_of_facades` |
| 12–17 | Two facade normals |
| 18–23 | Two normalized reflection points |
| 24–26 | Three segment lengths divided by 600 m |
| 27–30 | Ordered incidence/outgoing angle cosines |
| 31–32 | Finite-reflection-entry flags |
| 33–34 | Reflection-law flags |
| 35–37 | Segment-clearance flags |
| 38–39 | Direct-path and double-reflection flags |

Unavailable reflection entries and unused slots are zero padded. The effective valid mask is `(feature[0] > 0.5) and (feature[9] > 0.5)`. The named `FIELDS` list in the source is authoritative for individual coordinates.

## 4. Network structure and decoding

### Shared geometric branch

Each 40-dimensional descriptor passes through `Linear(40,64) -> SiLU -> Linear(64,64) -> SiLU`. The resulting 64-dimensional embedding is concatenated with the six normalized coordinates. A **shared** candidate MLP has dimensions `70 -> 128 -> 128 -> 9`, with SiLU after its two hidden layers.

Each output contains one existence logit, one gain scalar, one delay residual scalar, and two three-component direction residual vectors. Thus nine raw network outputs represent a logit plus a path whose final physical representation has gain, delay, and four angles. These are not nine independent physical path parameters.

For raw gain output `a_g`, the decoded gain is `20*a_g - 100` dB. The delay residual is `50 ns * tanh(a_tau)`, added to the geometric delay and clipped to `[0,tau_max]`. Each direction residual component is `0.2*tanh(a_u)`, added to its geometric direction. The z component is then folded into the required hemisphere (arrival positive, departure negative), bounded away from zero by `1e-8`, and the vector normalized with a norm floor of `1e-8`. Invalid-candidate logits are set to `-1e4`.

### Free-path branch

Masked mean and maximum pooling aggregate the valid 64-dimensional candidate embeddings. If there are no valid candidates, the pooled context is zero. Concatenating both pools and the six coordinates produces 134 inputs. The free MLP is `134 -> 128 -> 128 -> 9`, again with two SiLU hidden layers.

This branch produces **one additional proposal**, not one proposal per geometric candidate and not an arbitrary number of missing paths. Its gain uses the same affine scaling. Its delay is `tau_max * sigmoid(a_tau)` without a candidate-specific anchor. For directions, x/y components are predicted directly; a signed softplus z component enforces the canonical hemisphere before normalization. There is no attention selection or geometry/free mixing gate in this M3 architecture.

The two branches jointly produce 65 pre-filter proposals. Final decoded sets contain at most `K_max=3` paths. The numerical widths, residual bounds, and committee size are also summarized in Section IV-A of the manuscript.

## 5. Set supervision and the equivalent BCE form

Let `pi` be an injective assignment from measured components to distinct proposals. Set `y_i^pi=1` exactly for assigned proposals, and zero otherwise. For `p_i = sigmoid(gamma_i)`, the paper writes

$$\mathcal L_m=\min_\pi\left\{\sum_{\ell=1}^{\widetilde K_m}d_z^2(\widehat z_{\pi(\ell)},\widetilde z_{m\ell})+\lambda_p\sum_i\operatorname{BCE}(y_i^\pi,p_i)\right\}.$$

The BCE term is evaluated for all proposals; the assignment minimizes the **combined** parameter and existence cost. Since

$$\operatorname{BCE}(y,\operatorname{sigmoid}(\gamma))=\operatorname{softplus}(\gamma)-y\gamma,$$

the same objective is

$$\lambda_p\sum_i\operatorname{softplus}(\gamma_i)+\min_\pi\sum_\ell\left[d_z^2(\widehat z_{\pi(\ell)},\widetilde z_{m\ell})-\lambda_p\gamma_{\pi(\ell)}\right].$$

The executable implementation retains this stable logit/softplus form. The paper's BCE notation is an algebraic rewriting, not a new loss or a rerun. Training does not apply hard inference thresholds before matching, and the loss is not simply the evaluation GOSPA metric.

For at most three measured components, an optimal injective assignment can use each component's three lowest-cost proposals: at most two of those can be occupied by other components. The code enumerates the `3^3` combinations, rejects collisions for active labels, and masks padded label entries. An empty valid label contributes only the existence penalty. Failed extractions are excluded from the learning data; valid empty observations are retained.

## 6. Replay, warm starts, and committee output

`replay_weights` mixes uniform-frame sampling and cardinality-conditioned spatial-cell balancing with equal weights. The spatial branch first draws a measured cardinality at its empirical frequency, then an occupied cell uniformly within that cardinality, and then a frame uniformly from the selected cell. In symbols, the weight of frame `i` in cardinality group `k` and cell `c` is

$$w_i=\frac{0.5}{N}+0.5\,\frac{N_k/N}{C_k\,n_c},$$

where `C_k` is the number of occupied cells in the cardinality group and `n_c` its cell occupancy. Consequently the empirical cardinality marginal is preserved. Cell construction is delegated to `support.cell_keys`. Sampling is with replacement. Subsequent updates retain network weights, optimizer states, and member-specific random streams, rather than restarting the learner.

At inference, `decoding.py` computes probabilities and keeps proposals at or above the threshold (0.5 in the reported setup). It processes them in descending probability order and suppresses a proposal when a retained proposal is within 5 ns and within 5 degrees in **both** endpoint directions. It stops after three distinct retained proposals and sorts the resulting set by gain. Suppression does not average path parameters.

For each query, three independently initialized/trained committee members produce path sets. The final output is the member's **entire** set with the smallest sum of squared GOSPA distances to the other members. This is per-query set-medoid selection, not coordinate-wise averaging and not one globally selected member for all queries.

## 7. Acquisition and evaluation boundary

A round is a finite sounding window, not a complete survey of the 6D domain. The formal experiment uses one common initial 1800-s window followed by seven acquisition windows, with continued learning on accumulated valid data. The initial scan is fixed from the geometry and motion constraints; it does not require a trained predictor.

The paper's Cooperative policy uses history-based joint-link coverage novelty. Its coverage score is `1 - historical_coverage`; it does not use committee disagreement or the held-out final CKM error. The feasible beam planner, motion primitives, and return checks are implemented in the planning modules linked above. Measured labels update the learning set; all probe attempts update the position history. Held-out reference sets remain evaluation-only.

Other policies and earlier model variants coexist in this repository. Their uncertainty scores, attention mechanisms, or gates must not be substituted into the published coverage-only candidate-level configuration when interpreting the paper or its figures.
