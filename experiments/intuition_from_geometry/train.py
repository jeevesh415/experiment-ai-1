"""
Training loop for the Intuition-from-Geometry experiment.

Trains the energy-based manifold on concept hierarchies,
then tests whether geometric intuition emerges.
"""

import torch
import torch.nn.functional as F
import time
import json
import os
from concept_space import ConceptHierarchy
from energy_manifold import IntuitionManifold


def train(epochs=200, latent_dim=2048, lr=3e-4, batch_size=64, 
          device="cpu", save_every=50, log_every=10):
    """
    Train the Intuition Manifold.
    
    The training process:
    1. Encode concepts into latent space
    2. Decode back and measure reconstruction
    3. Shape the energy landscape so valid concepts sit in low-energy basins
    4. Train attractor dynamics so noisy inputs converge to clean concepts
    """
    print(f"Training Intuition Manifold")
    print(f"  Device: {device}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print()
    
    # Initialize
    hierarchy = ConceptHierarchy(dim=latent_dim)
    model = IntuitionManifold(input_dim=latent_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"  Concepts: {len(hierarchy.concepts)}")
    print(f"  Analogy pairs: {len(hierarchy.analogy_pairs)}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Training loop
    history = []
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        start_time = time.time()
        
        # Multiple batches per epoch
        batches_per_epoch = max(1, len(hierarchy.concepts) // batch_size)
        
        for _ in range(batches_per_epoch):
            batch = hierarchy.generate_training_batch(batch_size)
            concepts = batch["concepts"].to(device)
            parents = batch["parents"].to(device)
            children = batch["children"].to(device)
            partials = batch["partials"].to(device)
            
            # Forward pass
            x_recon, z, mean, logvar = model(concepts)
            
            # Compute loss
            loss, loss_dict = model.compute_loss(
                concepts, x_recon, z, mean, logvar,
                parent_x=parents, child_x=children, partial_x=partials
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss_dict)
        
        scheduler.step()
        
        # Aggregate losses
        avg_losses = {}
        for key in epoch_losses[0]:
            avg_losses[key] = sum(d[key] for d in epoch_losses) / len(epoch_losses)
        
        elapsed = time.time() - start_time
        history.append({"epoch": epoch, **avg_losses, "time": elapsed})
        
        # Logging
        if epoch % log_every == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{epochs} | "
                  f"Loss: {avg_losses['total']:.4f} | "
                  f"Recon: {avg_losses['recon']:.4f} | "
                  f"KL: {avg_losses['kl']:.4f} | "
                  f"Energy: {avg_losses['energy']:.4f} | "
                  f"Hier: {avg_losses['hierarchy']:.4f} | "
                  f"Attract: {avg_losses['attractor']:.4f} | "
                  f"({elapsed:.1f}s)")
        
        # Save best model
        if avg_losses['total'] < best_loss:
            best_loss = avg_losses['total']
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(os.path.dirname(__file__), 'best_model.pt'))
        
        # Periodic checkpoint
        if epoch % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': avg_losses['total'],
            }, os.path.join(os.path.dirname(__file__), f'checkpoint_epoch{epoch}.pt'))
    
    # Save training history
    with open(os.path.join(os.path.dirname(__file__), 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    
    return model, hierarchy, history


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, hierarchy, history = train(
        epochs=200,
        latent_dim=2048,
        lr=3e-4,
        batch_size=64,
        device=device,
        save_every=50,
        log_every=10,
    )
