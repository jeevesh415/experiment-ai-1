"""
Experiment 1: Energy-Based Geometric Completion

A small, reproducible experiment for the hypothesis:

    A learned energy landscape can act as an associative attractor,
    recovering a valid structured concept from a corrupted observation.

This is deliberately small. It does NOT claim general intelligence.
It measures one falsifiable capability: structured completion.

Design:
- Concepts are compositions of independent attributes.
- Training sees most attribute combinations; held-out combinations test
  compositional generalization.
- An energy model is trained to assign low energy to valid concepts and
  higher energy to corrupted/negative states.
- At inference, the observed latent is refined by gradient descent on energy.
- A nearest-prototype baseline receives the same corrupted observation.

Run:
    python experiment1.py --epochs 300 --device cpu

The script writes experiment1_results.json.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    seed: int = 7
    latent_dim: int = 32
    attributes: int = 4
    values_per_attribute: int = 6
    train_fraction: float = 0.75
    epochs: int = 300
    batch_size: int = 64
    lr: float = 2e-3
    noise_std: float = 0.45
    refinement_steps: int = 40
    refinement_lr: float = 0.08


class CompositionalWorld:
    """Synthetic world with factorized, compositional concepts."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        g = torch.Generator().manual_seed(cfg.seed)

        # Each attribute value owns a direction in latent space.
        raw = torch.randn(
            cfg.attributes,
            cfg.values_per_attribute,
            cfg.latent_dim,
            generator=g,
        )
        self.basis = F.normalize(raw, dim=-1)

        self.all_codes = [
            tuple(code)
            for code in torch.cartesian_prod(
                *[torch.arange(cfg.values_per_attribute) for _ in range(cfg.attributes)]
            ).tolist()
        ]

        self.rng.shuffle(self.all_codes)
        n_train = int(len(self.all_codes) * cfg.train_fraction)
        self.train_codes = self.all_codes[:n_train]
        self.test_codes = self.all_codes[n_train:]

    def encode(self, codes):
        rows = []
        for code in codes:
            z = torch.zeros(self.cfg.latent_dim)
            for a, value in enumerate(code):
                z = z + self.basis[a, value]
            rows.append(F.normalize(z, dim=-1))
        return torch.stack(rows)

    def sample(self, codes, n):
        chosen = [self.rng.choice(codes) for _ in range(n)]
        return self.encode(chosen), chosen


class EnergyModel(nn.Module):
    """Learned scalar energy E(z). Lower energy means more plausible state."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_negative(z: torch.Tensor) -> torch.Tensor:
    """Create hard negatives by mixing/shuffling dimensions between samples."""
    perm = torch.randperm(z.size(0), device=z.device)
    lam = torch.rand(z.size(0), 1, device=z.device) * 0.8 + 0.1
    mixed = F.normalize(lam * z + (1.0 - lam) * z[perm], dim=-1)
    return mixed


def train_energy(model, world, cfg, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    history = []

    train_z = world.encode(world.train_codes).to(device)

    for epoch in range(1, cfg.epochs + 1):
        idx = torch.randint(0, len(train_z), (cfg.batch_size,), device=device)
        positive = train_z[idx]
        negative = make_negative(positive)

        # Positive energy should be lower than negative energy by a margin.
        e_pos = model(positive)
        e_neg = model(negative)
        ranking = F.relu(0.5 + e_pos - e_neg).mean()

        # Keep energies numerically bounded.
        regularization = 1e-3 * (e_pos.square().mean() + e_neg.square().mean())
        loss = ranking + regularization

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch == 1 or epoch % 25 == 0:
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.detach().cpu()),
                    "positive_energy": float(e_pos.mean().detach().cpu()),
                    "negative_energy": float(e_neg.mean().detach().cpu()),
                }
            )

    return history


@torch.no_grad()
def nearest_prototype(observed, prototypes):
    observed = F.normalize(observed, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)
    sims = observed @ prototypes.T
    return sims.argmax(dim=-1)


def refine_by_energy(model, observed, steps, lr):
    """Optimize the latent state itself; the network weights remain frozen."""
    z = observed.detach().clone()
    for _ in range(steps):
        z.requires_grad_(True)
        energy = model(z).sum()
        grad = torch.autograd.grad(energy, z)[0]
        with torch.no_grad():
            z = F.normalize(z - lr * grad, dim=-1)
    return z.detach()


def evaluate(model, world, cfg, device, noise_std):
    test_z = world.encode(world.test_codes).to(device)
    train_z = world.encode(world.train_codes).to(device)

    noisy = F.normalize(
        test_z + noise_std * torch.randn_like(test_z),
        dim=-1,
    )

    baseline_idx = nearest_prototype(noisy, train_z)
    baseline_recon = train_z[baseline_idx]

    refined = refine_by_energy(
        model,
        noisy,
        steps=cfg.refinement_steps,
        lr=cfg.refinement_lr,
    )

    # A completion is correct only if it reaches the exact held-out concept.
    all_test_sims = refined @ test_z.T
    energy_idx = all_test_sims.argmax(dim=-1)

    baseline_train_sims = baseline_recon @ test_z.T
    baseline_test_idx = baseline_train_sims.argmax(dim=-1)

    energy_correct = (energy_idx == torch.arange(len(test_z), device=device)).float()
    baseline_correct = (
        baseline_test_idx == torch.arange(len(test_z), device=device)
    ).float()

    return {
        "held_out_concepts": len(world.test_codes),
        "noise_std": noise_std,
        "energy_completion_accuracy": float(energy_correct.mean().cpu()),
        "nearest_prototype_accuracy": float(baseline_correct.mean().cpu()),
        "energy_advantage": float(
            (energy_correct.mean() - baseline_correct.mean()).cpu()
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--noise", type=float, default=0.45)
    args = parser.parse_args()

    cfg = Config(epochs=args.epochs, noise_std=args.noise)
    seed_everything(cfg.seed)

    device = torch.device(args.device)
    world = CompositionalWorld(cfg)
    model = EnergyModel(cfg.latent_dim).to(device)

    history = train_energy(model, world, cfg, device)
    metrics = evaluate(model, world, cfg, device, args.noise)

    result = {
        "experiment": "energy_based_geometric_completion",
        "seed": cfg.seed,
        "device": str(device),
        "config": vars(cfg),
        "train_concepts": len(world.train_codes),
        "test_concepts": len(world.test_codes),
        "metrics": metrics,
        "training_history": history,
    }

    output = Path(__file__).with_name("experiment1_results.json")
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
