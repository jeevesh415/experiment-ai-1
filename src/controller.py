import torch
import torch.nn as nn
import torch.nn.functional as F

class FrontierController(nn.Module):
    """
    Hierarchical Active Inference Controller.
    Minimizes Variational Free Energy (VFE) to drive autonomous behavior.
    """
    def __init__(self, dim=4096):
        super().__init__()
        self.dim = dim
        
        # Generative Model: Predicts the next state
        self.transition_model = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        
        # Energy Head: Computes E = -ln P(o|s) - ln P(s)
        self.vfe_head = nn.Sequential(
            nn.Linear(dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )
        
        # ACT Confidence Head
        self.confidence = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def compute_vfe(self, state, observation_latent):
        """
        Calculates Variational Free Energy.
        VFE = Complexity - Accuracy
        """
        predicted_state = self.transition_model(state)
        accuracy = -F.mse_loss(predicted_state, observation_latent)
        complexity = self.vfe_head(state)
        
        return complexity - accuracy

    def monitor(self, state):
        """
        Tier 2/3 Monitoring: Confidence and Halting Signal.
        """
        conf = self.confidence(state)
        return conf

    def decay_energy(self, energy, t):
        """
        Global Energy Decay to prevent infinite meta-loops.
        """
        return energy * torch.exp(torch.tensor(-0.05 * t))
