# Experiment: Intuition from Space Geometry

## Hypothesis

In a well-structured continuous latent space, **intuition** — the ability to complete
partial patterns, recognize abstract relationships, and make cross-domain leaps —
emerges from the **geometry** of the space itself.

No explicit reasoning chain. No symbolic rules. Just the shape of the space guiding
inference toward high-density regions (attractor basins).

## Core Mechanism: Energy-Based Attractor Dynamics

The latent space is trained so that valid concepts sit in **low-energy basins** of an
energy function E(z). When the system receives partial or noisy input:

1. Map input → latent point z₀
2. Follow the energy gradient: z_{t+1} = z_t - η · ∇E(z_t)
3. Converge to z* = nearest attractor (valid concept)
4. Decode z* → completed output

This is geometric intuition: the space itself does the "thinking."

## What We're Testing

### Test 1: Pattern Completion
- Give the system a partial concept (e.g., "ani" → should complete to "animal")
- The energy landscape should pull the partial embedding toward the full concept

### Test 2: Abstraction Navigation
- Train concepts at multiple abstraction levels (dog → mammal → animal → living_thing)
- Test whether interpolation in latent space produces valid intermediate concepts

### Test 3: Cross-Domain Analogy
- Train on concept pairs (king:queen, man:woman, father:mother)
- Test whether the offset vector (king - man + woman) lands near "queen"

### Test 4: Noisy Input Recovery
- Add Gaussian noise to concept embeddings
- Measure how quickly and accurately the energy dynamics recover the original concept

## Architecture

- **Latent dimension:** 2048 (practical, scalable)
- **Encoder:** Maps concepts (text-like sequences) to latent vectors
- **Decoder:** Reconstructs concepts from latent vectors
- **Energy function:** Learned potential over the latent space
- **Training:** VAE reconstruction + contrastive organization + energy landscape shaping

## Running

```bash
python train.py          # Train the model
python test_intuition.py  # Run all intuition tests
```
