import torch
import torch.nn as nn
from core_engine import UnifiedManifold, FrontierLiquidSSM, ContinuousPatchEncoder
from logic_layer import FrontierVSALogic, AssociativeMemory, RenormalizationGroup
from controller import FrontierController

class FrontierCognitiveEngine(nn.Module):
    """
    The Frontier Unified Cognitive Engine.
    A single-brain architecture for autonomous reasoning.
    """
    def __init__(self, dim=4096):
        super().__init__()
        self.dim = dim
        self.manifold = UnifiedManifold(dim)
        self.encoder = ContinuousPatchEncoder(dim)
        self.ssm = FrontierLiquidSSM(dim)
        self.vsa = FrontierVSALogic(dim)
        self.memory = AssociativeMemory(dim)
        self.rg = RenormalizationGroup(dim)
        self.controller = FrontierController(dim)
        
        # Internal State
        self.h = torch.zeros(1, dim)
        self.user_latent = torch.randn(1, dim) # Persistent user character profile

    def perceive(self, raw_input):
        """Map sensory input into the manifold."""
        z = self.encoder(raw_input)
        return self.manifold.project(z)

    def autonomous_loop(self, sensory_latent, max_ponder=50):
        """
        Active Inference Pondering Loop.
        The brain evolves in M until VFE is minimized.
        """
        print(">>> INITIATING FRONTIER AUTONOMOUS LOOP <<<")
        
        for t in range(max_ponder):
            # 1. State Update (Liquid SSM)
            y, self.h = self.ssm(sensory_latent, self.h)
            self.h = self.manifold.project(self.h)
            
            # 2. Neuro-Symbolic Refinement (RG + VSA)
            self.h = self.rg(self.h)
            
            # 3. Active Inference (VFE)
            vfe = self.controller.compute_vfe(self.h, sensory_latent)
            decayed_vfe = self.controller.decay_energy(vfe, t)
            
            # 4. Meta-Monitoring (Confidence)
            conf = self.controller.monitor(self.h)
            
            if t % 5 == 0:
                print(f"[Step {t}] VFE: {decayed_vfe.item():.6f} | Conf: {conf.item():.4f}")
            
            # Emergent Halting: VFE below threshold or high confidence
            if decayed_vfe.item() < 0.001 or conf.item() > 0.98:
                print(f">>> INSIGHT EMERGED AT STEP {t} <<<")
                break
                
        return self.h

def execute_frontier():
    # Initialize Engine at 4096-dim Frontier scale
    engine = FrontierCognitiveEngine(dim=4096)
    
    # Simulate high-entropy multimodal input
    # (e.g. Fused Vision/Audio/Text frequency components)
    raw_input = torch.randn(1, 512)
    
    # Perception
    z = engine.perceive(raw_input)
    
    # Autonomous Reasoning
    final_state = engine.autonomous_loop(z)
    
    print("\n>>> EXECUTION COMPLETE <<<")
    print(f"Final State Norm: {torch.norm(final_state).item():.4f}")
    print(f"Manifold Point (first 5 dims): {final_state[0, :5].detach().numpy()}")

if __name__ == "__main__":
    execute_frontier()
