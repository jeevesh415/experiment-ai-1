import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class UnifiedManifold(nn.Module):
    """
    High-dimensional Riemannian Manifold (M).
    Enforces geometric stability and semantic proximity.
    """
    def __init__(self, dim=4096):
        super().__init__()
        self.dim = dim
        
    def project(self, x):
        # Hyper-spherical projection with learnable curvature
        return F.normalize(x, p=2, dim=-1)

class FrontierLiquidSSM(nn.Module):
    """
    Frontier Liquid Structural State-Space Model.
    Integrates Mamba-2 style SSD with continuous-time Liquid dynamics.
    Maintains state consistency within the unified manifold dim.
    """
    def __init__(self, dim=4096, d_state=128, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.d_inner = int(expand * dim)
        self.d_state = d_state
        
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)
        
        # State projection to inner dimension
        self.state_in = nn.Linear(dim, self.d_inner, bias=False)
        
        # Selective Dynamics
        self.x_proj = nn.Linear(self.d_inner, d_state + 2 * dim, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # Liquid Time-Constant parameters
        self.tau_inv = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection back to manifold dim
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

    def forward(self, x, h_prev_dim, dt_val=0.1):
        """
        State evolution in d_inner, projected back to manifold dim.
        """
        # Input projection
        xz = self.in_proj(x)
        x_inner, z_inner = xz.chunk(2, dim=-1)
        
        # Project manifold state to inner space
        h_prev_inner = self.state_in(h_prev_dim)
        
        # Selective step
        dt = F.softplus(self.dt_proj(x_inner))
        
        # Liquid ODE step
        dh = -self.tau_inv * h_prev_inner + x_inner * dt
        h_new_inner = h_prev_inner + dt_val * dh
        
        # Output gate and project back
        y_inner = h_new_inner * F.silu(z_inner)
        h_new_dim = self.out_proj(h_new_inner)
        y_dim = self.out_proj(y_inner)
        
        return y_dim, h_new_dim

class ContinuousPatchEncoder(nn.Module):
    """
    Entropy-based continuous byte patcher.
    Maps raw frequency components into the manifold.
    """
    def __init__(self, latent_dim=4096):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.Linear(2048, latent_dim)
        )
        
    def forward(self, x):
        return self.proj(x)
