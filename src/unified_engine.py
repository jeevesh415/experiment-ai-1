import torch
import torch.nn as nn
from core_engine import UnifiedManifold, LiquidSSMLayer, ContinuousEncoder
from logic_layer import VSALogic, RenormalizationLayer
from controller import MetaCognitiveController

class SingleBrainEngine(nn.Module):
    """
    The Unified First-Principles Cognitive Engine.
    Fuses all modalities into a single continuous latent manifold.
    """
    def __init__(self, dim=2048):
        super().__init__()
        self.dim = dim
        self.manifold = UnifiedManifold(dim)
        self.encoder = ContinuousEncoder(256, dim)
        self.working_memory = LiquidSSMLayer(dim)
        self.logic = VSALogic(dim)
        self.renorm = RenormalizationLayer(dim)
        self.controller = MetaCognitiveController(dim)
        
        # Persistent states
        self.h = torch.zeros(1, dim)
        self.user_profile = torch.randn(1, dim) # Simulated user character
        self.goal = torch.zeros(1, dim)

    def process_sensory_input(self, raw_input, modality="text"):
        """
        Maps raw input directly into the manifold.
        """
        # Simulated frequency-domain patching
        latent = self.encoder(raw_input)
        return self.manifold.project(latent)

    def think(self, input_trajectory, max_ponder_steps=20):
        """
        The Autonomous Reasoning Loop.
        Continuous evolution in M guided by Tier 2 and Tier 3.
        """
        print(f"--- Starting Autonomous Pondering ---")
        
        for step in range(max_ponder_steps):
            # 1. Tier 1: Process input and update working memory
            self.h = self.working_memory(input_trajectory, self.h)
            self.h = self.manifold.project(self.h)
            
            # 2. Neuro-Symbolic Renormalization
            self.h = self.renorm(self.h)
            
            # 3. Tier 2: Monitor confidence
            confidence = self.controller.tier2_monitor(self.h)
            
            # 4. Tier 3: Compute Energy Alignment
            energy = self.controller.compute_energy(self.h, self.user_profile, self.goal)
            aligned_energy = self.controller.tier3_align(energy, step)
            
            print(f"Step {step+1}: Energy={aligned_energy.item():.4f}, Confidence={confidence.item():.4f}")
            
            # Halting condition: Low energy OR High confidence
            if confidence.item() > 0.95 or aligned_energy.item() < 0.05:
                print(f"--- Brain Reached Insight at Step {step+1} ---")
                break
                
        return self.h

def run_demonstration():
    engine = SingleBrainEngine(dim=2048)
    
    # 1. Simulate Cross-Modal Input
    # A vector representing 'Vision: Dog' + 'Audio: Bark'
    vision_input = torch.randn(1, 256)
    audio_input = torch.randn(1, 256)
    
    # Encode into the unified manifold
    z_vision = engine.process_sensory_input(vision_input, modality="vision")
    z_audio = engine.process_sensory_input(audio_input, modality="audio")
    
    # FUSE into a single brain state via VSA Bundling
    fused_state = engine.logic.bundle([z_vision, z_audio])
    
    # 2. Start Thinking Loop
    final_insight = engine.think(fused_state)
    
    print("\nFinal Latent State (Unified Manifold Point):")
    print(final_insight[:, :10]) # Print first 10 dims
    print("...")

if __name__ == "__main__":
    run_demonstration()
