# Experiment 1 — Energy-Based Geometric Completion

## Question

Can a learned energy landscape recover a **held-out compositional concept** from a corrupted observation?

This is a deliberately small, falsifiable experiment. It is not a claim of AGI or consciousness.

## Setup

Each synthetic concept is a composition of four attributes:

```
concept = attribute_1 + attribute_2 + attribute_3 + attribute_4
```

There are 6 values per attribute, giving 1296 possible combinations.

Only 75% of combinations are available during training. The remaining 25% are held out.

At test time:

1. A held-out concept is corrupted with Gaussian noise.
2. The energy model receives the corrupted vector.
3. Its parameters remain frozen.
4. The latent vector itself is iteratively moved downhill:
   `z <- normalize(z - eta * grad E(z))`
5. The resulting state is classified against the complete candidate dictionary.

A matched feed-forward denoiser is trained on the same noisy training distribution and evaluated on the same held-out combinations.

## Why this matters

The important test is **compositional generalization**.

The model cannot simply memorize the exact held-out concept during training.

The energy hypothesis predicts that the learned landscape can create useful basins that extend to unseen combinations.

## Metrics

- **completion accuracy** — exact recovery of the held-out concept
- **target cosine** — continuous similarity to the true target
- **energy advantage** — energy model accuracy minus baseline accuracy

## Run

From this directory:

```bash
python experiment1.py --epochs 300 --device cpu
```

For CUDA:

```bash
python experiment1.py --epochs 300 --device cuda
```

The experiment writes:

```
experiment1_results.json
```

## Scientific interpretation

A positive result is only evidence for this specific hypothesis under this synthetic distribution.

A negative result is useful too: it tells us the current energy formulation does not provide the proposed generalization.

The next stage should add:

- multiple random seeds
- noise sweeps
- matched parameter-count baselines
- ablations of the energy dynamics
- latent trajectory measurements
- statistical confidence intervals
- a genuinely sequential environment

Do not label a result "emergent intelligence" without these controls.
