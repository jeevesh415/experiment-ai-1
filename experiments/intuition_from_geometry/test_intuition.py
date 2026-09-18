"""
Test Geometric Intuition.

Tests whether the trained latent space exhibits:
1. Pattern completion (partial → full concept)
2. Abstraction navigation (interpolation produces valid concepts)
3. Cross-domain analogy (offset vectors transfer)
4. Noisy input recovery (converges to correct concept)

These are the empirical tests of whether "intuition from geometry" works.
"""

import torch
import torch.nn.functional as F
import os
import json
from concept_space import ConceptHierarchy
from energy_manifold import IntuitionManifold


def load_model(latent_dim=2048, device="cpu"):
    """Load the best trained model."""
    model = IntuitionManifold(input_dim=latent_dim, latent_dim=latent_dim).to(device)
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'best_model.pt')
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state'])
        print(f"Loaded model from epoch {checkpoint['epoch']} (loss: {checkpoint['loss']:.4f})")
    else:
        print("WARNING: No trained model found. Using random weights.")
    
    model.eval()
    return model


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    return F.cosine_similarity(a, b, dim=-1).mean().item()


def find_nearest_concept(z, hierarchy, model):
    """Find the nearest concept to a latent vector."""
    with torch.no_grad():
        # Encode all concepts
        all_vecs = hierarchy.embedding_matrix
        z_all, _, _ = model.encode(all_vecs)
        
        # Find nearest by cosine similarity
        sims = F.cosine_similarity(z.unsqueeze(0), z_all, dim=-1)
        best_idx = sims.argmax().item()
        best_sim = sims[best_idx].item()
        
        return hierarchy.concepts[best_idx], best_sim


def test_pattern_completion(model, hierarchy, device="cpu"):
    """
    Test 1: Pattern Completion
    
    Give partial/noisy concept embeddings and check if the energy dynamics
    converge to the correct concept.
    """
    print("\n" + "="*60)
    print("TEST 1: Pattern Completion (Partial → Full)")
    print("="*60)
    
    test_cases = [
        ("golden_retriever", 0.5),   # 50% noise
        ("red_rose", 0.3),           # 30% noise
        ("emperor_penguin", 0.7),    # 70% noise — harder
        ("sugar_maple", 0.4),        # 40% noise
        ("orca", 0.6),               # 60% noise
    ]
    
    results = []
    
    for concept, noise_level in test_cases:
        concept_vec = hierarchy.get_concept_vector(concept).unsqueeze(0).to(device)
        
        # Create noisy version
        noisy = concept_vec + torch.randn_like(concept_vec) * noise_level
        noisy = F.normalize(noisy, p=2, dim=-1)
        
        # Run intuition (attractor dynamics)
        with torch.no_grad():
            completed, z_converged, steps = model.intuit(noisy.to(device))
        
        # Check result
        predicted, sim = find_nearest_concept(z_converged.squeeze(0), hierarchy, model)
        correct = predicted == concept
        
        # Also check direct encoding similarity
        direct_sim = cosine_sim(completed, concept_vec)
        
        results.append({
            "concept": concept,
            "noise": noise_level,
            "predicted": predicted,
            "correct": correct,
            "similarity": sim,
            "reconstruction_sim": direct_sim,
            "steps": steps,
        })
        
        status = "✓" if correct else "✗"
        print(f"  {status} {concept:20s} | noise={noise_level:.1f} | "
              f"→ {predicted:20s} | sim={sim:.3f} | recon={direct_sim:.3f} | "
              f"steps={steps}")
    
    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    avg_sim = sum(r["similarity"] for r in results) / len(results)
    print(f"\n  Accuracy: {accuracy:.1%} | Avg similarity: {avg_sim:.3f}")
    
    return results


def test_abstraction_navigation(model, hierarchy, device="cpu"):
    """
    Test 2: Abstraction Navigation
    
    Interpolate between specific and general concepts in latent space.
    If the space is well-organized, intermediate points should map to
    valid intermediate concepts.
    """
    print("\n" + "="*60)
    print("TEST 2: Abstraction Navigation (Interpolation)")
    print("="*60)
    
    test_chains = [
        ("golden_retriever", "living_thing"),
        ("red_rose", "plant"),
        ("emperor_penguin", "animal"),
        ("sugar_maple", "living_thing"),
    ]
    
    results = []
    
    for specific, general in test_chains:
        z_specific, _, _ = model.encode(
            hierarchy.get_concept_vector(specific).unsqueeze(0).to(device))
        z_general, _, _ = model.encode(
            hierarchy.get_concept_vector(general).unsqueeze(0).to(device))
        
        # Interpolate at multiple alpha values
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        chain_results = []
        
        for alpha in alphas:
            z_interp = (1 - alpha) * z_specific + alpha * z_general
            z_interp = F.normalize(z_interp, p=2, dim=-1)
            
            predicted, sim = find_nearest_concept(z_interp.squeeze(0), hierarchy, model)
            chain_results.append({
                "alpha": alpha,
                "predicted": predicted,
                "similarity": sim,
            })
        
        # Check if interpolation produces a valid abstraction chain
        expected_chain = hierarchy.get_abstraction_chain(specific)
        
        print(f"\n  {specific} → {general}:")
        print(f"  Expected chain: {' → '.join(expected_chain)}")
        print(f"  Interpolation:")
        
        for cr in chain_results:
            marker = "│" if cr["alpha"] not in [0.0, 1.0] else "●"
            print(f"    {marker} α={cr['alpha']:.2f} → {cr['predicted']:20s} (sim={cr['similarity']:.3f})")
        
        results.append({
            "specific": specific,
            "general": general,
            "chain": chain_results,
            "expected_chain": expected_chain,
        })
    
    return results


def test_analogy(model, hierarchy, device="cpu"):
    """
    Test 3: Cross-Domain Analogy
    
    Test whether offset vectors transfer: a - b + c ≈ d
    This tests whether the space captures abstract relationships
    as consistent geometric directions.
    """
    print("\n" + "="*60)
    print("TEST 3: Cross-Domain Analogy (a:b :: c:d)")
    print("="*60)
    
    analogy_batch = hierarchy.generate_analogy_batch(len(hierarchy.analogy_pairs))
    if analogy_batch is None:
        print("  No analogy pairs available.")
        return []
    
    a_vecs, b_vecs, c_vecs, d_vecs = analogy_batch
    a_vecs = a_vecs.to(device)
    b_vecs = b_vecs.to(device)
    c_vecs = c_vecs.to(device)
    d_vecs = d_vecs.to(device)
    
    results = []
    
    with torch.no_grad():
        # Encode all vectors
        z_a, _, _ = model.encode(a_vecs)
        z_b, _, _ = model.encode(b_vecs)
        z_c, _, _ = model.encode(c_vecs)
        z_d, _, _ = model.encode(d_vecs)
        
        # Compute analogy: a - b + c should be near d
        z_analogy = z_a - z_b + z_c
        z_analogy = F.normalize(z_analogy, p=2, dim=-1)
        
        # Check similarity to target d
        direct_sim = cosine_sim(z_analogy, z_d)
        
        # Find nearest concept to the analogy result
        for i in range(len(z_analogy)):
            predicted, sim = find_nearest_concept(z_analogy[i], hierarchy, model)
            
            # Get the original pair info
            pair = hierarchy.analogy_pairs[i]
            
            results.append({
                "pair": f"{pair[0]}:{pair[1]} :: {pair[2]}:{pair[3]}",
                "predicted": predicted,
                "target": pair[3],
                "correct": predicted == pair[3],
                "similarity": sim,
            })
            
            status = "✓" if predicted == pair[3] else "✗"
            print(f"  {status} {pair[0]}:{pair[1]} :: {pair[2]}:{pair[3]}")
            print(f"      → predicted: {predicted} (sim={sim:.3f})")
    
    accuracy = sum(1 for r in results if r["correct"]) / len(results) if results else 0
    print(f"\n  Analogy accuracy: {accuracy:.1%}")
    
    return results


def test_noisy_recovery(model, hierarchy, device="cpu"):
    """
    Test 4: Noisy Input Recovery
    
    Add increasing amounts of noise to concept embeddings and measure
    how accurately the energy dynamics recover the original concept.
    Tests the "robustness" of the attractor basins.
    """
    print("\n" + "="*60)
    print("TEST 4: Noisy Input Recovery (Robustness)")
    print("="*60)
    
    noise_levels = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    test_concepts = ["golden_retriever", "red_rose", "emperor_penguin", 
                     "blue_whale", "sugar_maple", "macaw"]
    
    results = []
    
    for noise in noise_levels:
        correct = 0
        total_sims = []
        
        for concept in test_concepts:
            concept_vec = hierarchy.get_concept_vector(concept).unsqueeze(0).to(device)
            noisy = concept_vec + torch.randn_like(concept_vec) * noise
            noisy = F.normalize(noisy, p=2, dim=-1)
            
            with torch.no_grad():
                completed, z_converged, steps = model.intuit(noisy)
            
            predicted, sim = find_nearest_concept(z_converged.squeeze(0), hierarchy, model)
            if predicted == concept:
                correct += 1
            total_sims.append(sim)
        
        accuracy = correct / len(test_concepts)
        avg_sim = sum(total_sims) / len(total_sims)
        
        results.append({
            "noise": noise,
            "accuracy": accuracy,
            "avg_similarity": avg_sim,
        })
        
        bar = "█" * int(accuracy * 20)
        print(f"  σ={noise:.1f} | Accuracy: {accuracy:.0%} {bar:20s} | Avg sim: {avg_sim:.3f}")
    
    return results


def run_all_tests(latent_dim=2048, device="cpu"):
    """Run all intuition tests and save results."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         INTUITION FROM GEOMETRY — TEST SUITE                ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    model = load_model(latent_dim, device)
    hierarchy = ConceptHierarchy(dim=latent_dim)
    
    all_results = {}
    
    all_results["pattern_completion"] = test_pattern_completion(model, hierarchy, device)
    all_results["abstraction_navigation"] = test_abstraction_navigation(model, hierarchy, device)
    all_results["analogy"] = test_analogy(model, hierarchy, device)
    all_results["noisy_recovery"] = test_noisy_recovery(model, hierarchy, device)
    
    # Save results
    results_path = os.path.join(os.path.dirname(__file__), 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {results_path}")
    print(f"{'='*60}")
    
    return all_results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_all_tests(latent_dim=2048, device=device)
