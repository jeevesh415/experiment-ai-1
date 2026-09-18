"""
Synthetic Concept Hierarchy Generator.

Generates structured concept data with:
- Multi-level abstraction hierarchies (specific → general)
- Cross-domain analogy pairs
- Partial/noisy variants for testing completion

The data is designed so that the latent space MUST organize concepts
geometrically to minimize reconstruction error — forcing abstraction
and analogy to emerge as geometric properties.
"""

import torch
import torch.nn.functional as F
import random
import math


class ConceptHierarchy:
    """
    A tree of concepts at multiple abstraction levels.
    
    Level 0: Specific instances (golden_retriever, tabby_cat, ...)
    Level 1: Categories (dog, cat, bird, ...)
    Level 2: Super-categories (mammal, reptile, ...)
    Level 3: Domains (animal, plant, ...)
    Level 4: Root (living_thing)
    """
    
    def __init__(self, dim=2048, seed=42):
        self.dim = dim
        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        
        # Build the hierarchy
        self.tree = {
            "living_thing": {
                "animal": {
                    "mammal": {
                        "dog": ["golden_retriever", "labrador", "poodle", "bulldog"],
                        "cat": ["tabby_cat", "persian_cat", "siamese_cat", "bengal_cat"],
                        "horse": ["arabian_horse", "mustang", "clydesdale", "pony"],
                        "whale": ["blue_whale", "humpback", "orca", "narwhal"],
                    },
                    "bird": {
                        "eagle": ["bald_eagle", "golden_eagle", "harpy_eagle"],
                        "parrot": ["macaw", "cockatoo", "budgie"],
                        "penguin": ["emperor_penguin", "king_penguin", "adelie_penguin"],
                    },
                    "fish": {
                        "shark": ["great_white", "hammerhead", "whale_shark"],
                        "salmon": ["atlantic_salmon", "chinook", "sockeye"],
                    },
                },
                "plant": {
                    "tree": {
                        "oak": ["red_oak", "white_oak", "live_oak"],
                        "pine": ["scots_pine", "white_pine", "ponderosa_pine"],
                        "maple": ["sugar_maple", "red_maple", "japanese_maple"],
                    },
                    "flower": {
                        "rose": ["red_rose", "white_rose", "yellow_rose"],
                        "tulip": ["red_tulip", "purple_tulip", "yellow_tulip"],
                        "orchid": ["phalaenopsis", "cattleya", "dendrobium"],
                    },
                },
            }
        }
        
        # Analogy pairs for cross-domain testing
        self.analogy_pairs = [
            ("dog", "puppy", "cat", "kitten"),
            ("dog", "bark", "cat", "meow"),
            ("oak", "acorn", "pine", "cone"),
            ("eagle", "nest", "penguin", "colony"),
            ("red_rose", "love", "white_rose", "purity"),
            ("golden_retriever", "fetch", "tabby_cat", "hunt"),
            ("blue_whale", "ocean", "arabian_horse", "desert"),
        ]
        
        # Build flat concept list and hierarchy info
        self.concepts = []
        self.concept_to_level = {}
        self.concept_to_parent = {}
        self.concept_to_children = {}
        self._flatten_tree(self.tree, None, 0)
        
        # Generate concept embeddings (learned later, but start with structured init)
        self._init_embeddings()
    
    def _flatten_tree(self, node, parent, level):
        """Recursively flatten the tree into a concept list."""
        if isinstance(node, dict):
            for concept, children in node.items():
                self.concepts.append(concept)
                self.concept_to_level[concept] = level
                self.concept_to_parent[concept] = parent
                if parent:
                    self.concept_to_children.setdefault(parent, []).append(concept)
                self._flatten_tree(children, concept, level + 1)
        elif isinstance(node, list):
            for concept in node:
                self.concepts.append(concept)
                self.concept_to_level[concept] = level
                self.concept_to_parent[concept] = parent
                if parent:
                    self.concept_to_children.setdefault(parent, []).append(concept)
    
    def _init_embeddings(self):
        """
        Initialize concept embeddings with geometric structure.
        
        Key insight: concepts at the same level should be roughly orthogonal,
        while parent-child relationships should be along consistent directions.
        This gives the latent space a head start on organizing hierarchically.
        """
        n = len(self.concepts)
        self.embeddings = {}
        
        # Generate a random orthogonal basis
        basis = torch.randn(n, self.dim)
        basis = F.normalize(basis, p=2, dim=-1)
        
        for i, concept in enumerate(self.concepts):
            level = self.concept_to_level[concept]
            parent = self.concept_to_parent[concept]
            
            if parent is None:
                # Root: center of the space
                self.embeddings[concept] = basis[i] * 2.0
            else:
                # Child: parent embedding + small perturbation + level scaling
                parent_emb = self.embeddings[parent]
                # Closer to parent at higher levels (more abstract = tighter cluster)
                spread = 1.0 / (level + 1)
                noise = basis[i] * spread
                self.embeddings[concept] = parent_emb + noise
        
        # Stack into a tensor
        self.embedding_matrix = torch.stack([self.embeddings[c] for c in self.concepts])
        self.embedding_matrix = F.normalize(self.embedding_matrix, p=2, dim=-1)
        
        # Rebuild dict with normalized embeddings
        for i, concept in enumerate(self.concepts):
            self.embeddings[concept] = self.embedding_matrix[i]
    
    def get_concept_vector(self, concept):
        """Get the embedding vector for a concept."""
        return self.embeddings[concept]
    
    def get_parent(self, concept):
        """Get the parent concept (one level up in abstraction)."""
        return self.concept_to_parent.get(concept)
    
    def get_children(self, concept):
        """Get child concepts (one level down in specificity)."""
        return self.concept_to_children.get(concept, [])
    
    def get_siblings(self, concept):
        """Get sibling concepts (same parent)."""
        parent = self.concept_to_parent.get(concept)
        if parent is None:
            return []
        return [c for c in self.concept_to_children.get(parent, []) if c != concept]
    
    def get_abstraction_chain(self, concept):
        """Get the full abstraction chain from concept to root."""
        chain = [concept]
        current = concept
        while self.concept_to_parent.get(current) is not None:
            current = self.concept_to_parent[current]
            chain.append(current)
        return chain
    
    def generate_training_batch(self, batch_size=32):
        """
        Generate a training batch with:
        1. Random concepts and their embeddings (for reconstruction)
        2. Parent-child pairs (for hierarchy learning)
        3. Partial concepts with their completions (for intuition training)
        """
        # Random concepts
        indices = torch.randint(0, len(self.concepts), (batch_size,))
        concept_vecs = self.embedding_matrix[indices]
        
        # Parent-child pairs
        parent_indices = []
        child_indices = []
        for _ in range(batch_size):
            concept = self.concepts[random.randint(0, len(self.concepts) - 1)]
            parent = self.concept_to_parent.get(concept)
            if parent is not None:
                parent_indices.append(self.concepts.index(parent))
                child_indices.append(self.concepts.index(concept))
            else:
                parent_indices.append(self.concepts.index(concept))
                child_indices.append(self.concepts.index(concept))
        
        parent_vecs = self.embedding_matrix[parent_indices]
        child_vecs = self.embedding_matrix[child_indices]
        
        # Partial → complete pairs (noise-based)
        partial_vecs = concept_vecs + torch.randn_like(concept_vecs) * 0.3
        partial_vecs = F.normalize(partial_vecs, p=2, dim=-1)
        
        return {
            "concepts": concept_vecs,
            "indices": indices,
            "parents": parent_vecs,
            "children": child_vecs,
            "partials": partial_vecs,
            "targets": concept_vecs,  # partials should complete to originals
        }
    
    def generate_analogy_batch(self, batch_size=16):
        """
        Generate analogy test batches.
        Returns (a, b, c, d) where a:b :: c:d
        """
        pairs = []
        for _ in range(batch_size):
            if self.analogy_pairs:
                a, b, c, d = self.rng.choice(self.analogy_pairs)
                pairs.append((
                    self.embeddings[a], self.embeddings[b],
                    self.embeddings[c], self.embeddings[d]
                ))
        
        if not pairs:
            return None
            
        a_vecs = torch.stack([p[0] for p in pairs])
        b_vecs = torch.stack([p[1] for p in pairs])
        c_vecs = torch.stack([p[2] for p in pairs])
        d_vecs = torch.stack([p[3] for p in pairs])
        
        return a_vecs, b_vecs, c_vecs, d_vecs


def print_hierarchy(hierarchy):
    """Print the concept hierarchy."""
    print(f"Total concepts: {len(hierarchy.concepts)}")
    print(f"Latent dimension: {hierarchy.dim}")
    print()
    
    def _print_node(concept, indent=0):
        level = hierarchy.concept_to_level[concept]
        children = hierarchy.concept_to_children.get(concept, [])
        print(f"{'  ' * indent}[L{level}] {concept}")
        for child in children:
            _print_node(child, indent + 1)
    
    _print_node("living_thing")


if __name__ == "__main__":
    h = ConceptHierarchy(dim=2048)
    print_hierarchy(h)
    
    print(f"\nAnalogy pairs: {len(h.analogy_pairs)}")
    for a, b, c, d in h.analogy_pairs:
        print(f"  {a}:{b} :: {c}:{d}")
    
    # Test batch generation
    batch = h.generate_training_batch(4)
    print(f"\nBatch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
