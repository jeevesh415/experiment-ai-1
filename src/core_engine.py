import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class UnifiedManifold(nn.Module):
    """
    The Unified Continuous Latent Manifold (M).
    A high-dimensional Riemannian space where all information resides.
    """
    def __init__(self, dim=2048):
        super().__init__()
        self.dim = dim
        
    def project(self, x):
        # Project onto the surface of a hypersphere to maintain stability
        return F.normalize(x, p=2, dim=-1)

class LiquidSSMLayer(nn.Module):
    """
    Liquid Structural State-Space Layer.
    Combines Mamba-style scaling with continuous-time Liquid Neural Network dynamics.
    Implements: dh/dt = -1/tau * h + f(x, h)
    """
    def __init__(self, dim=2048, d_state=64):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        
        # Continuous-time parameters
        self.tau_inv = nn.Parameter(torch.ones(dim)) # Inverse time constants
        self.A = nn.Parameter(torch.randn(dim, d_state) / math.sqrt(d_state))
        self.B = nn.Parameter(torch.randn(dim, d_state) / math.sqrt(d_state))
        self.C = nn.Parameter(torch.randn(d_state, dim) / math.sqrt(dim))
        
        self.input_proj = nn.Linear(dim, dim)
        self.state_proj = nn.Linear(d_state, dim)

    def forward(self, x, h_prev, dt=0.1):
        """
        Euler discretization of the Liquid ODE:
        h_new = h_prev + dt * (-tau_inv * h_prev + input_effect)
        """
        # Compute state transition
        # Project input into state space
        input_state = torch.matmul(x, self.A) 
        
        # ODE Step
        dh = -self.tau_inv * h_prev + torch.matmul(input_state, self.C)
        h_new = h_prev + dt * dh
        
        return h_new

class ContinuousEncoder(nn.Module):
    """
    Tokenless continuous encoder.
    Maps raw frequency-domain patches into the manifold.
    """
    def __init__(self, input_dim=256, latent_dim=2048):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, latent_dim)
        )
        
    def forward(self, x):
        return self.proj(x)
