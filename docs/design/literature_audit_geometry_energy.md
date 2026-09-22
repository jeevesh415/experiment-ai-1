# Literature Audit — Geometry / Energy / Attractor Direction

This is a working literature map, not a claim of exhaustive coverage.

## Established mechanisms we must not claim as new

### Energy-based models
Energy functions have long been used as learned compatibility landscapes,
with inference performed by optimization or sampling.

### Hopfield and associative-memory dynamics
Attractor-based retrieval from an energy landscape is classical. Modern
Hopfield networks substantially extend capacity and connect associative
retrieval with attention-like mechanisms.

### Continuous-time attractors
Continuous-time Hopfield-style memories and related dynamical systems remain
an active research direction.

### Compositional energy minimization
Oarga & Du, NeurIPS 2025, construct global energy landscapes by composing
energies of smaller subproblems and demonstrate generalization to larger
reasoning instances. This is directly relevant and is a mandatory baseline
for any future claim in this direction.

### Compositional generalization theory
There is active theoretical work identifying conditions under which neural
systems can generalize to unseen combinations of known components.

### Geometry and continuous dynamics
Geometric deep learning, Neural ODEs, continuous dynamical systems, and
manifold representations already provide extensive foundations for treating
learned representations as continuous state spaces.

## References

- Oarga & Du (2025), *Generalizable Reasoning through Compositional Energy
  Minimization*, NeurIPS 2025:
  https://papers.nips.cc/paper_files/paper/2025/file/7b22ed7325629fd4c041d47139410fd9-Paper-Conference.pdf

- Oarga & Du project/code page:
  https://alexoarga.github.io/compositional_reasoning/

- Santos et al. (2025), *Modern Hopfield Networks with Continuous-Time
  Memories*:
  https://arxiv.org/abs/2502.10122

- Montanari et al. (2026), *Energy-Based Dynamical Models for Neurocomputation,
  Learning, and Optimization*:
  https://arxiv.org/abs/2604.05042

- Li (2025), *A Theoretical Analysis of Compositional Generalization in Neural
  Networks: A Necessary and Sufficient Condition*:
  https://arxiv.org/abs/2505.02627

- Geometric Deep Learning:
  https://arxiv.org/abs/2104.13478

## Research rule

Do not call the architecture "novel" merely because several known mechanisms
are combined. A novelty claim requires a clearly specified mechanism,
demonstrable distinction from prior work, and empirical/theoretical evidence.
