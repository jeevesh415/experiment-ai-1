"""
Energy-Based Manifold: The Core of Geometric Intuition.

Architecture:
- Continuous VAE that maps concepts to/from a latent space
- Learned energy function E(z) over the latent space
- Attractor dynamics: z_{t+1} = z_t - η · ∇E(z_t)
- The energy landscape has basins around valid concepts

The key insight: intuition = falling into the nearest energy basin.
No explicit reasoning. Just geometry doing the work.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConceptEncoder(nn.Module):
    """
    Encodes concept vectors into latent space.
    Maps from concept embedding dim → latent dim via a learned projection.
    """
    def __init__(self, input_dim=2048, latent_dim=2048, hidden_dim=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim * 2),  # mean + logvar
        )
    
    def forward(self, x):
        h = self.net(x)
        mean, logvar = h.chunk(2, dim=-1)
        return mean, logvar


class ConceptDecoder(nn.Module):
    """
    Decodes latent vectors back to concept space.
    Maps from latent dim → concept embedding dim.
    """
    def __init__(self, latent_dim=2048, output_dim=2048, hidden_dim=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, z):
        return self.net(z)


class EnergyFunction(nn.Module):
    """
    Learned energy function E(z) over the latent space.
    
    Low energy = valid concept region (attractor basin)
    High energy = invalid/interstitial region
    
    Architecture: Multi-scale potential with local and global components.
    - Local: Fine-grained basins around specific concepts
    - Global: Overall structure that keeps concepts in a bounded region
    """
    def __init__(self, latent_dim=2048, hidden_dim=4096, n_heads=8):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Global energy: ensures boundedness
        self.global_energy = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Local energy heads: capture fine-grained basin structure
        self.n_heads = n_heads
        self.local_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dim // n_heads),
                nn.GELU(),
                nn.Linear(hidden_dim // n_heads, 1),
            )
            for _ in range(n_heads)
        ])
        
        # Learnable mixing between global and local
        self.mix = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, z):
        """
        Compute energy E(z). Lower = better.
        z: [batch, latent_dim]
        returns: [batch, 1]
        """
        global_e = self.global_energy(z)
        
        local_es = [head(z) for head in self.local_heads]
        local_e = torch.stack(local_es, dim=-1).mean(dim=-1)
        
        mix = torch.sigmoid(self.mix)
        return mix * global_e + (1 - mix) * local_e
    
    def gradient(self, z):
        """
        Compute ∇E(z) — the direction of steepest energy increase.
        Used for attractor dynamics: move in the NEGATIVE gradient direction.
        """
        z_grad = z.detach().requires_grad_(True)
        energy = self(z_grad).sum()
        energy.backward()
        return z_grad.grad


class AttractorDynamics:
    """
    Energy-based attractor dynamics for intuition.
    
    Given a starting point z₀ in latent space:
    1. Compute energy gradient ∇E(z)
    2. Step downhill: z ← z - η · ∇E(z)
    3. Repeat until convergence (energy minimum)
    
    The converged point z* is the "intuitive completion" of the input.
    """
    def __init__(self, energy_fn, step_size=0.01, max_steps=100, 
                 convergence_threshold=1e-4):
        self.energy_fn = energy_fn
        self.step_size = step_size
        self.max_steps = max_steps
        self.convergence_threshold = convergence_threshold
    
    def step(self, z):
        """Single attractor step."""
        grad = self.energy_fn.gradient(z)
        return z - self.step_size * grad
    
    def converge(self, z, track_trajectory=False):
        """
        Run attractor dynamics until convergence.
        Returns the converged point and optionally the full trajectory.
        """
        trajectory = [z.detach()] if track_trajectory else None
        
        current = z
        for t in range(self.max_steps):
            grad = self.energy_fn.gradient(current)
            next_z = current - self.step_size * grad
            
            if track_trajectory:
                trajectory.append(next_z.detach())
            
            # Check convergence
            delta = (next_z - current).norm(dim=-1).mean()
            if delta < self.convergence_threshold:
                break
            
            current = next_z
        
        if track_trajectory:
            return current, trajectory, t + 1
        return current, t + 1


class IntuitionManifold(nn.Module):
    """
    The complete Intuition-from-Geometry model.
    
    Components:
    - Encoder: maps concepts → latent space
    - Decoder: maps latent space → concepts
    - Energy function: defines the landscape
    - Attractor dynamics: the "intuition" mechanism
    
    Training objectives:
    1. Reconstruction: encoder-decoder cycle preserves concepts
    2. Energy organization: valid concepts have low energy
    3. Hierarchy: parent-child pairs are geometrically close
    4. Attractor convergence: noisy inputs converge to clean concepts
    """
    def __init__(self, input_dim=2048, latent_dim=2048):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.encoder = ConceptEncoder(input_dim, latent_dim)
        self.decoder = ConceptDecoder(latent_dim, input_dim)
        self.energy = EnergyFunction(latent_dim)
        self.dynamics = AttractorDynamics(self.energy)
    
    def encode(self, x):
        """Encode concepts to latent space (with reparameterization)."""
        mean, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z, mean, logvar
    
    def decode(self, z):
        """Decode latent vectors to concept space."""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass: encode → decode."""
        z, mean, logvar = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z, mean, logvar
    
    def intuit(self, partial_input, track=False):
        """
        The core intuition mechanism.
        
        1. Encode partial/noisy input to latent space
        2. Run attractor dynamics to find nearest energy basin
        3. Decode the result
        
        This is what "intuition from geometry" means:
        the space itself completes the pattern.
        """
        z, _, _ = self.encode(partial_input)
        
        if track:
            z_converged, trajectory, steps = self.dynamics.converge(z, track_trajectory=True)
            result = self.decode(z_converged)
            return result, z_converged, trajectory, steps
        else:
            z_converged, steps = self.dynamics.converge(z)
            result = self.decode(z_converged)
            return result, z_converged, steps
    
    def compute_loss(self, x, x_recon, z, mean, logvar, 
                     parent_x=None, child_x=None, partial_x=None):
        """
        Multi-objective loss:
        1. Reconstruction loss (VAE)
        2. KL divergence (structured latent space)
        3. Energy organization (valid concepts = low energy)
        4. Hierarchy loss (parent-child geometric proximity)
        5. Attractor convergence loss (noisy → clean)
        """
        # 1. Reconstruction
        recon_loss = F.mse_loss(x_recon, x)
        
        # 2. KL divergence (keeps latent space organized)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / x.shape[0]
        kl_loss = kl_loss / self.latent_dim  # normalize
        
        # 3. Energy organization: encoded concepts should have LOW energy
        energy_at_concepts = self.energy(z).mean()
        
        # 4. Hierarchy loss: parent-child should be close in latent space
        hierarchy_loss = torch.tensor(0.0, device=x.device)
        if parent_x is not None and child_x is not None:
            z_parent, _, _ = self.encode(parent_x)
            z_child, _, _ = self.encode(child_x)
            # Children should be close to parents (but not identical)
            hierarchy_loss = F.mse_loss(z_child, z_parent.detach()) * 0.1
        
        # 5. Attractor convergence: partials should converge to originals
        attractor_loss = torch.tensor(0.0, device=x.device)
        if partial_x is not None:
            z_partial, _, _ = self.encode(partial_x)
            z_converged, _ = self.dynamics.converge(z_partial)
            # Converged point should be close to original encoding
            attractor_loss = F.mse_loss(z_converged, z.detach())
        
        total = (recon_loss + 
                 0.1 * kl_loss + 
                 0.01 * energy_at_concepts + 
                 hierarchy_loss + 
                 0.5 * attractor_loss)
        
        return total, {
            "recon": recon_loss.item(),
            "kl": kl_loss.item(),
            "energy": energy_at_concepts.item(),
            "hierarchy": hierarchy_loss.item(),
            "attractor": attractor_loss.item(),
            "total": total.item(),
        }
