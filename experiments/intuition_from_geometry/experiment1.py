"""
Experiment 1: Energy-Based Geometric Completion

Falsifiable question:
    Can a learned energy landscape recover held-out compositional concepts
    from corrupted observations better than a matched feed-forward denoiser?

The test set contains attribute combinations never shown during training.
Inference is allowed to compare against the complete candidate dictionary
only for evaluation; neither model sees test labels during training.

Run:
    python experiment1.py --epochs 300 --device cpu
"""

from __future__ import annotations

import argparse
import json
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
    """Synthetic world in which concepts are compositions of attributes."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        g = torch.Generator().manual_seed(cfg.seed)

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
            for attribute, value in enumerate(code):
                z = z + self.basis[attribute, value]
            rows.append(F.normalize(z, dim=-1))
        return torch.stack(rows)


class EnergyModel(nn.Module):
    """Scalar energy E(z); valid states should lie in low-energy basins."""

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


class Denoiser(nn.Module):
    """Matched neural baseline trained only on training combinations."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, dim),
        )

    def forward(self, z):
        return F.normalize(self.net(z), dim=-1)


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_negative(z: torch.Tensor) -> torch.Tensor:
    """Create hard negatives by interpolating between different concepts."""
    perm = torch.randperm(z.size(0), device=z.device)
    lam = torch.rand(z.size(0), 1, device=z.device) * 0.8 + 0.1
    return F.normalize(lam * z + (1.0 - lam) * z[perm], dim=-1)


def train_energy(model, world, cfg, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    train_z = world.encode(world.train_codes).to(device)
    history = []

    for epoch in range(1, cfg.epochs + 1):
        idx = torch.randint(0, len(train_z), (cfg.batch_size,), device=device)
        positive = train_z[idx]
        negative = make_negative(positive)

        e_pos = model(positive)
        e_neg = model(negative)

        ranking = F.relu(0.5 + e_pos - e_neg).mean()
        regularization = 1e-3 * (e_pos.square().mean() + e_neg.square().mean())
        loss = ranking + regularization

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch == 1 or epoch % 25 == 0:
            history.append({
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "positive_energy": float(e_pos.mean().detach().cpu()),
                "negative_energy": float(e_neg.mean().detach().cpu()),
            })

    return history


def train_denoiser(model, world, cfg, device):
    """Train the baseline without exposing held-out combinations."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    clean = world.encode(world.train_codes).to(device)

    for _ in range(cfg.epochs):
        idx = torch.randint(0, len(clean), (cfg.batch_size,), device=device)
        target = clean[idx]
        noisy = F.normalize(target + cfg.noise_std * torch.randn_like(target), dim=-1)

        prediction = model(noisy)
        loss = 1.0 - F.cosine_similarity(prediction, target, dim=-1).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def refine_by_energy(model, observed, steps, lr):
    """Move the latent state downhill on the learned energy landscape."""
    z = observed.detach().clone()

    for _ in range(steps):
        z.requires_grad_(True)
        energy = model(z).sum()
        grad = torch.autograd.grad(energy, z)[0]
        with torch.no_grad():
            z = F.normalize(z - lr * grad, dim=-1)

    return z.detach()


@torch.no_grad()
def classify_by_candidate_dictionary(z, all_concepts):
    """Evaluation-only nearest-neighbour classification."""
    z = F.normalize(z, dim=-1)
    candidates = F.normalize(all_concepts, dim=-1)
    return (z @ candidates.T).argmax(dim=-1)


def evaluate(energy, denoiser, world, cfg, device, noise_std):
    train_z = world.encode(world.train_codes).to(device)
    test_z = world.encode(world.test_codes).to(device)
    all_z = world.encode(world.all_codes).to(device)

    noisy = F.normalize(
        test_z + noise_std * torch.randn_like(test_z),
        dim=-1,
    )

    refined = refine_by_energy(
        energy,
        noisy,
        steps=cfg.refinement_steps,
        lr=cfg.refinement_lr,
    )

    denoised = denoiser(noisy)

    energy_idx = classify_by_candidate_dictionary(refined, all_z)
    denoiser_idx = classify_by_candidate_dictionary(denoised, all_z)

    # Map the held-out target code to its position in the full dictionary.
    target_indices = torch.tensor(
        [world.all_codes.index(code) for code in world.test_codes],
        device=device,
    )

    energy_correct = (energy_idx == target_indices).float()
    denoiser_correct = (denoiser_idx == target_indices).float()

    # Also report reconstruction similarity; this is continuous rather than
    # an all-or-nothing exact-class metric.
    energy_sim = F.cosine_similarity(refined, test_z, dim=-1).mean()
    denoiser_sim = F.cosine_similarity(denoised, test_z, dim=-1).mean()

    return {
        "held_out_concepts": len(world.test_codes),
        "noise_std": noise_std,
        "energy_completion_accuracy": float(energy_correct.mean().cpu()),
        "denoiser_completion_accuracy": float(denoiser_correct.mean().cpu()),
        "energy_advantage": float(
            (energy_correct.mean() - denoiser_correct.mean()).cpu()
        ),
        "energy_target_cosine": float(energy_sim.cpu()),
        "denoiser_target_cosine": float(denoiser_sim.cpu()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--noise", type=float, default=0.45)
    args = parser.parse_args()

    cfg = Config(epochs=args.epochs, noise_std=args.noise)
    seed_everything(cfg.seed)

    device = torch.device(args.device)
    world = CompositionalWorld(cfg)

    energy = EnergyModel(cfg.latent_dim).to(device)
    denoiser = Denoiser(cfg.latent_dim).to(device)

    history = train_energy(energy, world, cfg, device)
    train_denoiser(denoiser, world, cfg, device)

    metrics = evaluate(
        energy,
        denoiser,
        world,
        cfg,
        device,
        args.noise,
    )

    result = {
        "experiment": "energy_based_geometric_completion",
        "hypothesis": (
            "Energy-based attractor dynamics can recover held-out "
            "compositional concepts from corruption."
        ),
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
