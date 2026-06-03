# Tiny Qwen Subspace Observatory

## Spectral Geometry, Subspace Dynamics, ICA, and Representation Evolution in Transformer Training

---

# Overview

Tiny Qwen Subspace Observatory is an experimental research platform for studying how internal representations emerge, evolve, stabilize, reorganize, and interact during Transformer training.

Most language model research focuses on metrics such as:

* training loss
* validation loss
* perplexity
* downstream task performance

While these metrics answer the question:

> "How well is the model performing?"

they reveal very little about:

> "What internal structures are being created during learning?"

This repository addresses that gap.

Instead of treating training as a sequence of scalar optimization steps, we analyze training as a process of geometric self-organization.

The observatory continuously monitors:

* singular value spectra
* stable rank evolution
* effective dimensionality
* subspace formation
* subspace stabilization
* spectral transitions
* Independent Component Analysis (ICA)
* cross-layer feature organization
* component lifetimes
* component superposition
* optimizer-induced geometric biases

The result is a framework that allows researchers to observe how a Transformer gradually constructs its internal representation geometry.

---

# Core Research Questions

The project is built around several fundamental questions:

### When do useful internal structures appear?

Do representations emerge gradually?

Or do they appear through discrete reorganization events?

---

### When do representations stabilize?

Does representation convergence occur at the same time as loss convergence?

Or do internal geometries stabilize long before optimization finishes?

---

### How do different optimizers shape geometry?

Do AdamW and Muon learn the same internal structures?

Or do they create fundamentally different geometric organizations?

---

### Can activation subspaces be observed directly?

Can we identify independent computational networks inside the model?

Can we track their birth, growth, stabilization, and disappearance?

---

# Main Hypothesis

The central hypothesis of this project is:

[
\text{Feature Discovery}
\rightarrow
\text{Subspace Formation}
\rightarrow
\text{Subspace Stabilization}
\rightarrow
\text{Weight Refinement}
]

This differs from the conventional optimization-centric view.

We hypothesize that:

[
\text{Subspace Convergence}
<
\text{Loss Convergence}
]

meaning that the geometric structure of the model stabilizes before the optimization process is complete.

---

# Model Architecture

The repository uses a compact Qwen-style decoder-only Transformer.

Configuration:

| Parameter      | Value        |
| -------------- | ------------ |
| Layers         | 4            |
| Hidden Size    | 256          |
| FFN Dimension  | 1024         |
| Vocabulary     | 30,000       |
| Dataset        | WikiText-2   |
| Context Length | Configurable |

The small size allows dense diagnostic instrumentation while remaining computationally practical.

---

# Supported Optimizers

## AdamW

The standard Transformer baseline.

Characteristics:

* adaptive learning rates
* decoupled weight decay
* strong empirical performance

---

## Muon

A matrix-aware optimizer based on orthogonalized momentum updates.

Key ideas:

* matrix-valued optimization
* Newton–Schulz orthogonalization
* preservation of spectral diversity
* improved matrix conditioning

Muon treats weight matrices as geometric objects rather than collections of independent scalar parameters.

---

# Spectral Diagnostics

## Singular Value Spectrum

For every tracked matrix:

* attention projections
* FFN up projections
* FFN gate projections
* FFN down projections

the singular value decomposition is computed.

[
W = U \Sigma V^T
]

Tracked metrics include:

* largest singular value
* top-k singular values
* spectral gap
* cumulative spectral energy
* effective rank

These measurements provide a direct view of how information becomes distributed across matrix directions.

---

## Stable Rank

Stable rank is defined as:

[
\text{StableRank}(W)
====================

\frac{|W|_F^2}
{|W|_2^2}
]

Unlike algebraic rank, stable rank is robust to noise.

Interpretation:

### Low Stable Rank

Energy is concentrated in a small number of dominant directions.

The matrix behaves approximately low-rank.

### High Stable Rank

Energy is distributed across many directions.

The representation utilizes a larger portion of available capacity.

---

## Top-k Energy

Measures how much matrix energy is concentrated inside the leading singular directions.

[
E_k
===

\frac{\sum_{i=1}^{k}\sigma_i^2}
{\sum_i \sigma_i^2}
]

High values indicate concentration.

Low values indicate distributed structure.

---

# Subspace Dynamics

The most important idea in the repository is that singular vectors define evolving geometric subspaces.

For the top-k singular vectors:

[
U_k
]

we track their movement through training.

---

## Grassmann Distance

Subspace drift is measured on the Grassmann manifold.

Given two subspaces:

[
U_k^{(t)}
]

and

[
U_k^{(t+\Delta)}
]

we compute their principal angles and derive a Grassmann distance.

Interpretation:

### Small Drift

The representation remains stable.

### Large Drift

The representation is reorganizing.

---

## Subspace Stabilization

A subspace is considered stabilized when drift remains below a threshold for a sustained period.

This creates a notion of:

> geometric convergence

that is separate from loss convergence.

---

# Event Detection

The observatory automatically identifies geometric events.

---

## Subspace Birth

Previously weak structures become detectable.

Possible interpretation:

* feature emergence
* circuit formation
* specialization

---

## Spectral Separation

Leading singular directions become significantly stronger than surrounding directions.

Possible interpretation:

* hierarchy formation
* dominant feature emergence

---

## Local Stabilization

Subspace motion decreases dramatically.

Possible interpretation:

* convergence of a representation

---

## Destabilization Spike

A previously stable subspace suddenly moves.

Possible interpretation:

* representation restructuring
* feature recombination
* optimizer-induced reorganization

These events are especially interesting when comparing AdamW and Muon.

---

# ICA Observatory

The repository contains a complete Independent Component Analysis framework.

The ICA system is inspired by fMRI analysis.

---

## Spatial ICA

For each checkpoint, FFN activations are collected from all Transformer blocks.

Let:

[
A_l
]

be the FFN activation matrix from layer (l).

These are concatenated:

[
X
=

[A_0 | A_1 | A_2 | A_3]
]

FastICA is then applied.

The result is a set of independent activation networks.

---

## Why ICA?

SVD identifies directions of variance.

ICA attempts to identify independent computational processes.

This makes ICA particularly suitable for studying:

* distributed representations
* feature circuits
* emergent subnetworks

---

## Component Count

The number of ICA components is estimated using:

* explained variance
* participation ratio

The resulting quantity:

[
n_{ICA}(t)
]

acts as a measure of activation complexity.

---

## Cross-Layer Bind Index

Measures how strongly a component spans multiple Transformer blocks.

Low values:

* local feature

High values:

* distributed network

This provides evidence for cross-layer computational organization.

---

## Component Lifetime

Components are matched across checkpoints.

Each component receives:

* birth time
* lifetime
* death time

This allows us to study representation persistence.

---

## Component Genealogy (Planned)

Future versions will track:

[
A
\rightarrow
A_1 + A_2
]

component splits

and

[
A+B
\rightarrow
C
]

component mergers.

This will create a true evolutionary model of representation dynamics.

---

# Superposition Analysis

Modern mechanistic interpretability suggests that many features may share neurons.

The observatory measures neuron-level superposition.

For neuron (j):

[
S_j
===

\frac{
(\sum_i w_{ij}^2)^2
}
{
\sum_i w_{ij}^4
}
]

Interpretation:

Low values:

* specialization

High values:

* feature sharing
* superposition

---

# Temporal ICA

Spatial ICA answers:

> Where does a component live?

Temporal ICA answers:

> When does it become active?

Temporal ICA operates on component trajectories across training.

Potential modes include:

* feature formation
* feature consolidation
* memorization
* refinement

---

# Token-Level Component Tracing

An experimental extension allows component activations to be measured during inference.

For token representation:

[
h_t
]

and ICA component:

[
c_k
]

activation is:

[
a_{k,t}
=======

h_t \cdot c_k
]

This allows visualization of:

* domain transitions
* syntax-sensitive features
* uncertainty responses
* representation switching

Example:

```text
Narrative prose
      ↓
Structured JSON
```

The transition can become visible directly in component activity maps.

---

# Main Experimental Findings

Current experiments indicate:

### Muon Produces Higher Stable Rank

Muon consistently maintains broader spectral support.

---

### Muon Produces Fewer ICA Components

Activation organization becomes more compact.

---

### Muon Stabilizes More Subspaces

A larger fraction of tracked subspaces converge.

---

### Muon Creates More Coherent Geometric Trajectories

Training states occupy a smaller number of geometric regimes.

---

### Better Geometry Does Not Necessarily Mean Better Validation Loss

Loss and geometry are distinct objectives.

This is one of the central conclusions of the project.

---

# Future Directions

## Multi-Seed Validation

Determine which effects remain statistically robust.

---

## Larger Models

Apply the framework to:

* Qwen 0.5B
* Qwen 1.5B
* Llama-family models

---

## Component Genealogy

Track births, deaths, splits, and mergers of computational components.

---

## Domain Shift Detection

Use component activations to identify:

* prose
* code
* JSON
* mathematical text
* tabular data

without explicit labels.

---

## Optimizer Geometry Research

Study how:

* AdamW
* Muon
* Shampoo
* K-FAC
* Sophia

shape internal representation geometry.

---

# References

* Bernstein & Newhouse — Muon
* Liu et al. — Understanding Muon Through Singular Value Spectra
* Chen et al. — Spectral Implicit Bias and Lion-K
* Fan et al. — Max-Margin Analysis of Muon
* Hyvärinen & Oja — Independent Component Analysis
* Edelman et al. — Geometry of the Grassmann Manifold
* Kornblith et al. — CKA
* Raghu et al. — SVCCA
* Pennington et al. — Random Matrix Theory and Deep Learning

---

**Status:** Active Research Platform

**Focus:** Transformer Geometry, Representation Dynamics, Spectral Learning, and Emergent Computational Structure.
