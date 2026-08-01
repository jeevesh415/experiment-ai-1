import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaCognitiveController(nn.Module):
    """
    3-Tier Meta-Cognitive Controller.
    Tier 1: Working System (Base processing)
    Tier 2: Meta System (Confidence & ACT)
    Tier 3: Super-Meta System (Long-term Alignment & Energy Minimization)
    """
    def __init__(self, dim=2048):
        super().__init__()
        self.dim = dim
        
        # Tier 2: Meta-monitoring
        self.confidence_head = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        # Tier 3: Energy Function Head
        # Represents the variational free energy E(s, u, g)
        self.energy_head = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def compute_energy(self, state, user_profile, goal):
        """
        Calculates the internal 'Free Energy' of the system.
        Higher energy = higher uncertainty/misalignment.
        """
        # Combine state, user, and goal into a single context
        context = state + user_profile + goal
        energy = self.energy_head(context)
        return energy

    def tier2_monitor(self, state):
        """
        Tier 2 checks if we should halt (ACT).
        """
        confidence = self.confidence_head(state)
        return confidence

    def tier3_align(self, energy, time_elapsed):
        """
        Tier 3 ensures long-term alignment and prevents meta-loops
        via Global Energy Decay.
        """
        # Decay energy over time to force a decision
        decay_factor = torch.exp(torch.tensor(-0.1 * time_elapsed))
        adjusted_energy = energy * decay_factor
        return adjusted_energy
