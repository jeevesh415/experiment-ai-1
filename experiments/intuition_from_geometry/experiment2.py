"""
Experiment 2: Geometry-as-Computation Benchmark

Research question:
    When novel combinations of known factors are never shown during training,
    does a learned continuous dynamical geometry provide useful inference
    beyond standard feed-forward denoising and explicit compositional baselines?

This benchmark is deliberately designed against known literature:
- feed-forward denoising
- additive/vector-space composition
- an energy-based attractor
- an explicit factorized energy baseline

No test combination is used for training. The candidate dictionary is used
only for evaluation.

The experiment is not a novelty claim. It is a falsification harness:
a proposed "geometric cognition" mechanism earns attention only if it
beats matched baselines on held-out compositions, under ablations.

Run:
    python experiment2.py --epochs 500 --seeds 3 --device cpu
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
    latent_dim: int = 48
    attributes: int = 5
    values_per_attribute: int = 5
    train_fraction: float = 0.70
    epochs: int = 500
    batch_size: int = 96
    lr: float = 2e-3
    noise_std: float = 0.35
    refine_steps: int = 50
    refine_lr: float = 0.05


def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class World:
    def __init__(self, cfg: Config):
        g = torch.Generator().manual_seed(cfg.seed)
        self.cfg = cfg
        self.basis = F.normalize(
            torch.randn(
                cfg.attributes, cfg.values_per_attribute, cfg.latent_dim,
                generator=g,
            ),
            dim=-1,
        )
        codes = [
            tuple(c)
            for c in torch.cartesian_prod(
                *[torch.arange(cfg.values_per_attribute)
                  for _ in range(cfg.attributes)]
            ).tolist()
        ]
        random.Random(cfg.seed).shuffle(codes)
        n = int(len(codes) * cfg.train_fraction)
        self.train_codes = codes[:n]
        self.test_codes = codes[n:]
        self.all_codes = codes

    def encode(self, codes):
        out = []
        for code in codes:
            z = sum(
                (self.basis[a, v] for a, v in enumerate(code)),
                torch.zeros(self.cfg.latent_dim),
            )
            out.append(F.normalize(z, dim=-1))
        return torch.stack(out)


class MLPDenoiser(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 160), nn.GELU(),
            nn.Linear(160, 160), nn.GELU(),
            nn.Linear(160, d),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class GlobalEnergy(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 160), nn.SiLU(),
            nn.Linear(160, 160), nn.SiLU(),
            nn.Linear(160, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FactorEnergy(nn.Module):
    """
    Explicit compositional baseline.

    Each factor gets its own scalar energy network. At inference, the total
    energy is the sum over factors. This is intentionally strong: if the
    geometric attractor cannot beat this, its claimed advantage is weak.
    """
    def __init__(self, d, attributes):
        super().__init__()
        self.factor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, 96), nn.Tanh(),
                nn.Linear(96, 1),
            )
            for _ in range(attributes)
        ])

    def forward(self, x):
        return torch.stack([m(x).squeeze(-1) for m in self.factor], dim=-1).sum(-1)


def corrupt(z, std):
    return F.normalize(z + std * torch.randn_like(z), dim=-1)


def train_denoiser(model, clean, cfg, device):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    for _ in range(cfg.epochs):
        idx = torch.randint(len(clean), (cfg.batch_size,), device=device)
        target = clean[idx]
        pred = model(corrupt(target, cfg.noise_std))
        loss = 1 - F.cosine_similarity(pred, target, dim=-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


def train_energy(model, clean, cfg, device):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    for _ in range(cfg.epochs):
        idx = torch.randint(len(clean), (cfg.batch_size,), device=device)
        pos = clean[idx]
        perm = torch.randperm(len(pos), device=device)
        lam = 0.1 + 0.8 * torch.rand(len(pos), 1, device=device)
        neg = F.normalize(lam * pos + (1 - lam) * pos[perm], dim=-1)
        ep, en = model(pos), model(neg)
        loss = F.relu(0.4 + ep - en).mean() + 1e-3 * (ep.square().mean() + en.square().mean())
        opt.zero_grad()
        loss.backward()
        opt.step()


def refine(model, x, steps, lr):
    z = x.detach().clone()
    for _ in range(steps):
        z.requires_grad_(True)
        e = model(z).sum()
        grad = torch.autograd.grad(e, z)[0]
        with torch.no_grad():
            z = F.normalize(z - lr * grad, dim=-1)
    return z.detach()


def accuracy(z, candidates, target_idx):
    idx = (F.normalize(z, dim=-1) @ F.normalize(candidates, dim=-1).T).argmax(-1)
    return float((idx == target_idx).float().mean())


@torch.no_grad()
def cosine(z, target):
    return float(F.cosine_similarity(z, target, dim=-1).mean())


def run(seed, args):
    cfg = Config(seed=seed, epochs=args.epochs, noise_std=args.noise)
    seed_all(seed)
    device = torch.device(args.device)
    world = World(cfg)

    train = world.encode(world.train_codes).to(device)
    test = world.encode(world.test_codes).to(device)
    candidates = world.encode(world.all_codes).to(device)
    target_idx = torch.tensor(
        [world.all_codes.index(c) for c in world.test_codes], device=device
    )

    denoiser = MLPDenoiser(cfg.latent_dim).to(device)
    global_energy = GlobalEnergy(cfg.latent_dim).to(device)
    factor_energy = FactorEnergy(cfg.latent_dim, cfg.attributes).to(device)

    train_denoiser(denoiser, train, cfg, device)
    train_energy(global_energy, train, cfg, device)
    train_energy(factor_energy, train, cfg, device)

    observed = corrupt(test, cfg.noise_std)
    denoised = denoiser(observed)
    global_refined = refine(global_energy, observed, cfg.refine_steps, cfg.refine_lr)
    factor_refined = refine(factor_energy, observed, cfg.refine_steps, cfg.refine_lr)

    # Identity baseline: noisy observation itself.
    results = {
        "identity": {
            "accuracy": accuracy(observed, candidates, target_idx),
            "cosine": cosine(observed, test),
        },
        "mlp_denoiser": {
            "accuracy": accuracy(denoised, candidates, target_idx),
            "cosine": cosine(denoised, test),
        },
        "global_energy": {
            "accuracy": accuracy(global_refined, candidates, target_idx),
            "cosine": cosine(global_refined, test),
        },
        "factorized_energy": {
            "accuracy": accuracy(factor_refined, candidates, target_idx),
            "cosine": cosine(factor_refined, test),
        },
    }
    return {
        "seed": seed,
        "train_combinations": len(world.train_codes),
        "held_out_combinations": len(world.test_codes),
        "metrics": results,
    }


def aggregate(runs):
    methods = runs[0]["metrics"].keys()
    out = {}
    for method in methods:
        acc = [r["metrics"][method]["accuracy"] for r in runs]
        sim = [r["metrics"][method]["cosine"] for r in runs]
        out[method] = {
            "accuracy_mean": sum(acc) / len(acc),
            "accuracy_std": math.sqrt(
                sum((x - sum(acc) / len(acc)) ** 2 for x in acc) / len(acc)
            ),
            "cosine_mean": sum(sim) / len(sim),
            "cosine_std": math.sqrt(
                sum((x - sum(sim) / len(sim)) ** 2 for x in sim) / len(sim)
            ),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--noise", type=float, default=0.35)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    runs = [run(7 + i, args) for i in range(args.seeds)]
    result = {
        "experiment": "geometry_as_computation_benchmark",
        "research_question": (
            "Does a learned continuous energy geometry provide information "
            "useful for unseen compositions beyond matched feed-forward and "
            "explicit compositional baselines?"
        ),
        "literature_position": (
            "This benchmark does not claim that energy landscapes, attractor "
            "dynamics, or compositional energy minimization are new. It tests "
            "whether a specific geometric mechanism adds measurable value."
        ),
        "runs": runs,
        "aggregate": aggregate(runs),
    }

    path = Path(__file__).with_name("experiment2_results.json")
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["aggregate"], indent=2))
    print(f"Results: {path}")


if __name__ == "__main__":
    main()
