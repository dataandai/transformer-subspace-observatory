# Transformer Subspace Observatory

**Reproducible diagnostics for comparing spectral, subspace, and activation-space dynamics during small Transformer training runs.**

This repository is an engineering playground for observing how different optimizers shape the internal geometry of a compact Transformer during training.

The main motivation is simple:

> Run the same small Transformer setup with AdamW and Muon, log internal geometry, and compare what changes.

This is not intended to be a benchmark paper, a mechanistic interpretability proof, or a claim that one optimizer is universally better than another. It is a reproducible measurement harness for generating hypotheses about optimizer-induced representation dynamics.

---

## What this is

Transformer training is usually monitored through scalar metrics such as:

- training loss
- validation loss
- perplexity
- downstream task performance

Those metrics are useful, but they do not show much about how the model's internal matrices and activation spaces change during training.

This repository adds a second view: geometry-first diagnostics.

It tracks:

- singular value spectra
- stable rank
- effective rank
- top-k spectral energy
- Grassmann subspace drift
- thresholded subspace events
- ICA-derived activation components
- component persistence across checkpoints
- neuron-level superposition proxies
- optimizer-induced differences between AdamW and Muon

The goal is to make these measurements easy to reproduce, compare, and extend.

---

## What this is not

This repository does **not** claim that:

- ICA components are proven circuits
- subspace events are proven feature births
- Muon is universally better than AdamW
- geometric metrics are better objectives than validation loss
- the observed effects are statistically robust across all seeds, model sizes, or datasets

The outputs should be treated as diagnostic signals and exploratory evidence, not final mechanistic explanations.

---

## Main use case

The intended workflow is:

1. Train a compact Qwen-style decoder-only Transformer.
2. Run the same configuration with different optimizers, especially AdamW and Muon.
3. Save checkpoints and diagnostic logs.
4. Compare the internal geometry of the training runs.
5. Use the results to form hypotheses for better-controlled experiments.

The project is designed to make small-scale optimizer comparison easy and reproducible.

---

## Core questions

The repository is built around practical measurement questions:

### Do AdamW and Muon produce different matrix spectra?

For example:

- Does one optimizer concentrate energy into fewer singular directions?
- Does one maintain broader spectral support?
- Do spectral gaps evolve differently?

### Do their dominant subspaces move differently?

For example:

- Do top-k singular subspaces stabilize earlier under one optimizer?
- Are there periods of rapid subspace drift?
- Are some layers more geometrically stable than others?

### Do their activation spaces differ?

For example:

- Do ICA-derived activation components appear more compact or more distributed?
- Are components more layer-local or cross-layer?
- Are component trajectories stable across checkpoints?

### Does geometry tell a different story from validation loss?

The repository is especially interested in cases where validation loss and geometric diagnostics disagree.

That disagreement is not treated as a failure. It is one of the main reasons to measure internal geometry in the first place.

---

## Model setup

The current experiments use a compact Qwen-style decoder-only Transformer.

Typical configuration:

| Parameter | Value |
|---|---:|
| Layers | 4 |
| Hidden size | 256 |
| FFN dimension | 1024 |
| Vocabulary size | 30,000 |
| Dataset | WikiText-2 |
| Context length | Configurable |

The model is intentionally small. This makes dense checkpointing and expensive diagnostic analysis practical.

The purpose is not to train a strong language model. The purpose is to create a controlled environment where optimizer-induced geometry can be inspected.

---

## Supported optimizers

### AdamW

AdamW is used as the standard Transformer optimizer baseline.

It provides:

- adaptive learning rates
- decoupled weight decay
- a strong and familiar reference point

AdamW is useful here because it gives a conventional baseline for comparing spectral and subspace dynamics.

### Muon

Muon is included as a matrix-aware optimizer based on orthogonalized momentum updates.

In this repository, Muon is interesting because it may affect matrix geometry differently from AdamW.

The comparison is not framed as:

> Muon is better.

It is framed as:

> Muon and AdamW may leave different geometric signatures during training.

The observatory exists to measure those signatures.

---

## Spectral diagnostics

The repository computes singular value diagnostics for tracked Transformer matrices, including:

- attention projection matrices
- FFN up projections
- FFN gate projections
- FFN down projections

For a weight matrix \(W\), the singular value decomposition is:

\[
W = U \Sigma V^\top
\]

Tracked quantities include:

- largest singular value
- top-k singular values
- spectral gap
- cumulative spectral energy
- stable rank
- effective rank

These diagnostics describe how weight energy is distributed across matrix directions.

---

## Stable rank

Stable rank is defined as:

\[
\mathrm{StableRank}(W) = \frac{\|W\|_F^2}{\|W\|_2^2}
\]

It is a soft rank measure.

A lower stable rank means that matrix energy is concentrated in fewer dominant directions.

A higher stable rank means that energy is distributed across more directions.

In this project, stable rank is used as a diagnostic proxy for spectral spread. It should not be interpreted as a direct measure of model quality.

---

## Top-k spectral energy

Top-k spectral energy measures how much matrix energy is contained in the leading singular directions:

\[
E_k = \frac{\sum_{i=1}^{k}\sigma_i^2}{\sum_i \sigma_i^2}
\]

High top-k energy indicates concentration in the leading directions.

Lower top-k energy indicates more distributed spectral support.

This is useful for comparing whether AdamW and Muon produce different patterns of matrix energy concentration.

---

## Subspace dynamics

The top-k singular vectors define a low-dimensional subspace.

For each checkpoint, the repository tracks the movement of these subspaces through training.

Given top-k subspaces at two checkpoints:

\[
U_k^{(t)}
\]

and

\[
U_k^{(t+\Delta)}
\]

the principal angles between them are used to compute a Grassmann-style subspace distance.

This gives a checkpoint-to-checkpoint measure of subspace drift.

---

## Grassmann drift

Grassmann drift is used as a proxy for how much a dominant singular subspace changes over time.

Small drift suggests that the tracked subspace is relatively stable.

Large drift suggests that the tracked subspace is changing substantially.

This does **not** automatically mean that a representation has converged or reorganized in a mechanistic sense. It only means that the measured top-k subspace moved less or more according to the chosen metric.

---

## Thresholded subspace events

The observatory can flag thresholded events in spectral and subspace trajectories.

Examples include:

### Subspace detection event

A previously weak or noisy tracked subspace becomes measurable under the chosen threshold.

Possible interpretation:

- a new dominant direction becomes visible
- spectral mass becomes less diffuse
- a tracked structure crosses a diagnostic threshold

### Spectral separation event

Leading singular directions separate from the surrounding spectrum.

Possible interpretation:

- stronger concentration in top directions
- emergence of a clearer spectral hierarchy
- optimizer-dependent spectral organization

### Local stabilization event

Subspace drift remains below a threshold for a sustained period.

Possible interpretation:

- the tracked top-k subspace has become locally stable
- the dominant directions are changing more slowly

### Destabilization spike

A previously stable tracked subspace shows a sudden increase in drift.

Possible interpretation:

- a temporary reorganization of dominant directions
- optimizer- or schedule-induced change
- checkpoint-level instability

These events are diagnostic labels, not proof of underlying causal mechanisms.

---

## ICA activation observatory

The repository includes an Independent Component Analysis workflow for FFN activations.

For each checkpoint, FFN activations are collected from Transformer blocks and concatenated into a shared activation matrix.

If \(A_l\) is the FFN activation matrix from layer \(l\), the concatenated activation matrix is:

\[
X = [A_0 \mid A_1 \mid A_2 \mid A_3]
\]

FastICA is then applied to estimate activation components.

These components are referred to as **ICA-derived activation components**.

They should not be treated as proven circuits. They are exploratory components extracted under ICA assumptions.

---

## Why use ICA?

SVD identifies directions of variance.

ICA attempts to find statistically independent components under a stronger modeling assumption.

In this repository, ICA is used as an exploratory lens on activation space.

It can help ask questions such as:

- Are activation components more local or distributed?
- Do component counts differ between optimizers?
- Do similar components persist across checkpoints?
- Do component trajectories change during training?

The output is useful for exploration, but it requires validation before making mechanistic claims.

---

## Component count

The number of ICA components can be estimated using criteria such as:

- explained variance
- participation ratio
- diagnostic thresholds

The resulting quantity:

\[
n_{\mathrm{ICA}}(t)
\]

is treated as a rough measure of activation-space complexity at checkpoint \(t\).

It is not a definitive count of model features.

---

## Cross-layer bind index

The cross-layer bind index measures how strongly an ICA-derived activation component spans multiple Transformer blocks.

Lower values suggest that a component is more layer-local.

Higher values suggest that a component is more distributed across layers.

This is useful for comparing whether AdamW and Muon produce different cross-layer activation patterns.

---

## Component persistence

Components can be matched across checkpoints to estimate persistence over training.

For each matched component, the observatory can track:

- first detected checkpoint
- last detected checkpoint
- approximate lifetime
- similarity to components at neighboring checkpoints

This is intended as a temporal diagnostic.

Terms such as "birth" and "death" should be understood as shorthand for thresholded component detection and disappearance, not literal mechanistic events.

---

## Superposition proxy

The repository includes neuron-level superposition-style diagnostics.

For neuron \(j\), a participation-style score can be computed as:

\[
S_j = \frac{\left(\sum_i w_{ij}^2\right)^2}{\sum_i w_{ij}^4}
\]

Lower values suggest concentration.

Higher values suggest broader participation across components or directions.

This is a proxy metric. It is useful for comparison, but it should not be overinterpreted as a direct measurement of feature superposition.

---

## Temporal component analysis

The repository can also examine how component activity changes over training.

This is useful for studying whether components:

- appear early or late
- persist across many checkpoints
- become more or less layer-distributed
- differ between AdamW and Muon runs

The purpose is to compare trajectories, not to claim a final theory of representation evolution.

---

## Token-level component tracing

An experimental extension projects token representations onto ICA-derived components during inference.

For token representation \(h_t\) and component \(c_k\), the component activation is:

\[
a_{k,t} = h_t \cdot c_k
\]

This can be used to visualize how component activity changes across a token sequence.

Possible uses include exploratory inspection of:

- domain transitions
- formatting changes
- structured versus unstructured text
- token-level activation patterns

This feature is experimental and should be interpreted cautiously.

---

## Current observations

Initial experiments suggest that, in the tested small-scale setup, AdamW and Muon can produce visibly different geometric diagnostics.

Observed differences may include:

- different stable-rank trajectories
- different top-k spectral energy patterns
- different subspace drift profiles
- different numbers of ICA-derived activation components
- different local stabilization patterns
- geometry/loss disagreement in some runs

These are observations from small experiments.

They should be treated as hypotheses until validated with:

- multiple seeds
- stronger baselines
- larger models
- additional datasets
- ablations over hyperparameters
- null or randomized controls

---

## Reproducibility goal

A major goal of the repository is to make optimizer comparison easy to rerun.

The ideal workflow is:

```bash
# Run an AdamW experiment
python train.py --optimizer adamw --seed 1 --run-name adamw_seed1

# Run a Muon experiment
python train.py --optimizer muon --seed 1 --run-name muon_seed1

# Compare diagnostics
python compare_runs.py \
  --run-a runs/adamw_seed1 \
  --run-b runs/muon_seed1 \
  --metrics stable_rank grassmann_drift topk_energy ica_components
```

The exact command interface may evolve, but the intended design is:

- same model
- same dataset
- same seed
- same checkpoint schedule
- same diagnostics
- different optimizer

That makes the comparison interpretable and easy to reproduce.

---

## Suggested interpretation discipline

The repository intentionally separates measurements from interpretation.

Recommended wording:

| Instead of saying | Prefer saying |
|---|---|
| feature discovery | activation component emergence |
| circuit formation | ICA-derived component structure |
| subspace birth | thresholded subspace detection |
| representation convergence | reduced top-k subspace drift |
| Muon learns better geometry | Muon produced different geometric diagnostics in this setup |
| independent computational network | ICA-derived activation component |
| evolutionary model | checkpoint-wise component tracking |

This keeps the project useful without overstating what the diagnostics prove.

---

## Why this repository exists

This project exists because optimizer comparisons often collapse into one scalar question:

> Which optimizer gets lower validation loss?

That question matters, but it is not the only useful question.

This repository asks a different set of engineering questions:

- What happens to the spectra of the learned matrices?
- How do dominant subspaces move during training?
- Do optimizers produce different activation-space decompositions?
- Are there diagnostic signals that appear before loss differences?
- Can geometry help debug, compare, or understand training runs?

The goal is not to replace loss curves.

The goal is to put more instruments on the training process.

---

## Future directions

### Multi-seed validation

Run the same comparison across multiple seeds to determine which effects are stable and which are run-specific.

### Larger models

Apply the same diagnostics to larger Transformer variants, if compute allows.

Possible targets:

- larger Qwen-style models
- Llama-family models
- other decoder-only architectures

### Additional optimizers

Extend comparison beyond AdamW and Muon.

Potential candidates:

- Shampoo
- K-FAC
- Sophia
- Lion
- other matrix-aware or curvature-aware optimizers

### Better controls

Add controls such as:

- randomized checkpoints
- shuffled activations
- null baselines
- frozen-model baselines
- hyperparameter sweeps

### Component matching improvements

Improve checkpoint-to-checkpoint component matching with more robust similarity metrics and uncertainty estimates.

### Cleaner experiment runner

Move from notebook-only experimentation toward a clearer command-line experiment runner.

The long-term goal is to make the repository easier for others to reproduce, inspect, and extend.

---

## References and related ideas

This project is influenced by work on:

- Muon and matrix-aware optimization
- singular value spectra during training
- spectral implicit bias
- Independent Component Analysis
- Grassmannian geometry
- CKA and SVCCA-style representation comparison
- random matrix theory in deep learning
- mechanistic interpretability and feature superposition

The repository uses these ideas as engineering tools for diagnostics, not as final proof of a mechanistic theory.

---

## Status

**Status:** experimental engineering playground

**Focus:** reproducible optimizer comparison, Transformer geometry diagnostics, spectral tracking, subspace drift, and activation-space analysis.

This repository is most useful as a small, inspectable testbed for asking:

> When AdamW and Muon train the same small Transformer, what changes inside the model?

