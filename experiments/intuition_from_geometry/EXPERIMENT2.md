# Experiment 2 — Geometry as Computation

## Why this replaces the original experiment

Energy landscapes, attractor dynamics, associative memory, and compositional
energy minimization are established research areas. In particular, Oarga &
Du (NeurIPS 2025) explicitly compose subproblem energy functions to generalize
to larger unseen reasoning problems.

Therefore this experiment makes a narrower, falsifiable claim:

> When novel combinations of known factors are withheld, does a learned
> continuous energy geometry provide useful inference beyond a matched
> feed-forward denoiser and an explicit factorized-energy baseline?

## Experimental controls

The synthetic world contains 5 attributes with 5 values each: 3,125 possible
combinations.

Only 70% of combinations are available during training. The remaining 30%
are never presented as complete concepts during optimization.

Four methods are compared:

1. **Identity** — corrupted observation without inference.
2. **MLP denoiser** — conventional feed-forward reconstruction.
3. **Global energy** — one learned scalar energy landscape plus gradient-flow
   inference.
4. **Factorized energy** — an intentionally strong compositional baseline
   whose energy is an explicit sum of factor-specific energies.

The complete candidate dictionary is used only after inference to measure
nearest-concept accuracy. Test labels never enter training.

## What would count as evidence

A useful geometric mechanism should show:

- improvement over the identity baseline;
- improvement over the matched MLP;
- ideally improvement over the factorized energy baseline;
- stability across multiple random seeds;
- a meaningful advantage that survives parameter matching and ablations.

A single successful run is **not** evidence of a new form of intelligence.

## Required next experiments

Before making any novelty claim:

- sweep corruption/noise levels;
- match parameter counts;
- vary the number of withheld compositions;
- remove gradient refinement and compare;
- replace the learned geometry with ordinary Euclidean interpolation;
- test larger attribute/value counts;
- use relational rather than purely additive factors;
- report confidence intervals across seeds;
- add published compositional-generalization baselines where practical.

## Literature position

Relevant established directions include:

- Hopfield/associative-memory attractor dynamics;
- modern continuous-time Hopfield memories;
- energy-based models and optimization-based inference;
- geometric/deep dynamical representations;
- compositional generalization theory;
- compositional energy minimization for reasoning.

See the repository literature note for references and links.

The point of this experiment is therefore not to relabel existing ideas as novel.
It is to identify a measurable property that survives strong prior-art
baselines.
