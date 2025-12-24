#!/usr/bin/env python3
"""
Vector Library - Save and Load Semantic Vectors

Persist computed semantic vectors for reuse without recomputation.
Save vectors from convergence analysis and apply them to new words.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from ollama_client import OllamaEmbeddings
from utils import cosine_similarity, find_closest_words


class VectorLibrary:
    """Manage saved semantic vectors."""

    def __init__(self, library_dir: str = ".vectors"):
        """
        Initialize vector library.

        Args:
            library_dir: Directory to store saved vectors
        """
        self.library_dir = Path(library_dir)
        self.library_dir.mkdir(exist_ok=True)

    def save_vector(
        self,
        name: str,
        vector: np.ndarray,
        word_pairs: List[Tuple[str, str]],
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Save a semantic vector to the library.

        Args:
            name: Name for this vector (e.g., "masculinity", "geography")
            vector: The computed vector to save
            word_pairs: The word pairs used to compute this vector
            metadata: Optional metadata (model, coherence score, etc.)

        Returns:
            Path to saved file
        """
        if metadata is None:
            metadata = {}

        data = {
            "name": name,
            "vector": vector.tolist(),
            "word_pairs": word_pairs,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata,
        }

        filename = self.library_dir / f"{name}.json"
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        print(f"✅ Saved vector '{name}' to {filename}")
        return str(filename)

    def load_vector(self, name: str) -> Tuple[np.ndarray, Dict]:
        """
        Load a semantic vector from the library.

        Args:
            name: Name of the vector to load

        Returns:
            Tuple of (vector, metadata)
        """
        filename = self.library_dir / f"{name}.json"

        if not filename.exists():
            raise FileNotFoundError(f"Vector '{name}' not found in library")

        with open(filename, "r") as f:
            data = json.load(f)

        vector = np.array(data["vector"])
        return vector, data

    def list_vectors(self) -> List[Dict]:
        """
        List all saved vectors in the library.

        Returns:
            List of vector metadata
        """
        vectors = []

        for filename in self.library_dir.glob("*.json"):
            try:
                with open(filename, "r") as f:
                    data = json.load(f)
                    vectors.append(
                        {
                            "name": data["name"],
                            "word_pairs": data["word_pairs"],
                            "created_at": data.get("created_at", "unknown"),
                            "metadata": data.get("metadata", {}),
                        }
                    )
            except Exception as e:
                print(f"⚠️  Error reading {filename}: {e}")

        return vectors

    def delete_vector(self, name: str):
        """Delete a vector from the library."""
        filename = self.library_dir / f"{name}.json"
        if filename.exists():
            filename.unlink()
            print(f"✅ Deleted vector '{name}'")
        else:
            print(f"❌ Vector '{name}' not found")

    def apply_vector(
        self,
        vector_name: str,
        words: List[str],
        client: OllamaEmbeddings,
        top_k: int = 5,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Apply a saved vector to new words.

        Args:
            vector_name: Name of the saved vector to apply
            words: Words to transform
            client: Ollama client for embeddings
            top_k: Number of closest words to return

        Returns:
            Dictionary mapping input words to their transformations
        """
        # Load vector
        vector, metadata = self.load_vector(vector_name)

        print(f"\n🔍 Applying '{vector_name}' vector to {len(words)} words...")
        print(f"   Vector computed from {len(metadata['word_pairs'])} pairs")

        # Get embeddings for input words
        embeddings = client.get_embeddings(words)
        word_to_embedding = {word: emb for word, emb in zip(words, embeddings)}

        # Get common words for comparison
        common_words = self._get_common_words(client)

        # Apply vector and find closest words
        results = {}
        for word in words:
            transformed = word_to_embedding[word] + vector
            closest = find_closest_words(
                transformed, common_words, top_k=top_k, exclude=[word]
            )
            results[word] = closest

        return results

    def _get_common_words(self, client: OllamaEmbeddings) -> Dict[str, np.ndarray]:
        """Get embeddings for common comparison words."""
        common_words = [
            # People
            "man",
            "woman",
            "king",
            "queen",
            "boy",
            "girl",
            "father",
            "mother",
            "brother",
            "sister",
            "uncle",
            "aunt",
            "prince",
            "princess",
            "actor",
            "actress",
            # Professions
            "doctor",
            "nurse",
            "teacher",
            "professor",
            "engineer",
            "scientist",
            "artist",
            "musician",
            "chef",
            "cook",
            "waiter",
            "waitress",
            # Places
            "paris",
            "france",
            "london",
            "england",
            "rome",
            "italy",
            "berlin",
            "germany",
            "tokyo",
            "japan",
            # Things
            "pizza",
            "pasta",
            "bread",
            "coffee",
            "tea",
            "car",
            "bike",
            "house",
            "apartment",
        ]

        embeddings = client.get_embeddings(common_words)
        return {word: emb for word, emb in zip(common_words, embeddings)}


def save_from_convergence():
    """
    Interactive: Save a vector from convergence analysis.
    """
    from vector_convergence import VectorConvergenceAnalyzer

    print("=" * 80)
    print("   💾 Save Vector from Convergence Analysis")
    print("=" * 80)

    # Get category
    print("\nChoose a semantic dimension:")
    print("  1. Gender (masculinity)")
    print("  2. Geography (capital-country)")
    print("  3. Size (big-small)")
    print("  4. Animal (adult-young)")
    print("  5. Custom")

    choice = input("\nEnter choice (1-5): ").strip()

    category_map = {
        "1": ("gender", "masculinity", [
            ("man", "woman"),
            ("king", "queen"),
            ("boy", "girl"),
            ("father", "mother"),
            ("brother", "sister"),
            ("uncle", "aunt"),
            ("prince", "princess"),
            ("actor", "actress"),
        ]),
        "2": ("geography", "geography_capital_country", [
            ("paris", "france"),
            ("london", "england"),
            ("rome", "italy"),
            ("berlin", "germany"),
            ("madrid", "spain"),
            ("tokyo", "japan"),
            ("beijing", "china"),
            ("moscow", "russia"),
        ]),
        "3": ("size", "size_big_small", [
            ("big", "small"),
            ("large", "tiny"),
            ("huge", "minuscule"),
            ("giant", "dwarf"),
            ("massive", "minute"),
            ("enormous", "microscopic"),
        ]),
        "4": ("animal", "animal_adult_young", [
            ("dog", "puppy"),
            ("cat", "kitten"),
            ("cow", "calf"),
            ("horse", "foal"),
            ("sheep", "lamb"),
            ("lion", "cub"),
            ("bird", "chick"),
        ]),
    }

    if choice == "5":
        # Custom word pairs
        category = "custom"
        default_name = "custom_vector"
        
        print("\n📝 Enter custom word pairs (format: word1-word2, one per line)")
        print("   Example: happy-sad")
        print("   Type 'done' when finished\n")
        
        word_pairs = []
        while True:
            pair_input = input(f"Pair {len(word_pairs) + 1}: ").strip()
            
            if pair_input.lower() == "done":
                break
            
            if "-" not in pair_input:
                print("   ⚠️  Invalid format. Use: word1-word2")
                continue
            
            parts = pair_input.split("-", 1)
            if len(parts) == 2:
                word1, word2 = parts[0].strip(), parts[1].strip()
                if word1 and word2:
                    word_pairs.append((word1, word2))
                    print(f"   ✓ Added: {word1} - {word2}")
                else:
                    print("   ⚠️  Both words must be non-empty")
            else:
                print("   ⚠️  Invalid format. Use: word1-word2")
        
        if not word_pairs:
            print("\n❌ No word pairs entered")
            return
        
        print(f"\n✓ Entered {len(word_pairs)} word pairs")
        
    elif choice in category_map:
        category, default_name, word_pairs = category_map[choice]
    else:
        print("❌ Invalid choice")
        return

    # Get name
    name = input(f"\nVector name [{default_name}]: ").strip() or default_name

    # Connect to Ollama
    try:
        client = OllamaEmbeddings()
        
        # Get dimensions and display model info
        test_emb = client.get_embedding("test")
        dimensions = test_emb.shape[0]
        
        print(f"\n📊 EMBEDDING MODEL: {client.model}")
        print(f"📏 Vector Dimensions: {dimensions}")
        print(f"🌐 Host: {client.host}")
    except (ConnectionError, ValueError) as e:
        print(f"❌ {e}")
        return

    # Analyze convergence
    print(f"\n🔍 Computing vector from {len(word_pairs)} pairs...")
    analyzer = VectorConvergenceAnalyzer(client)
    analyzer.add_word_pairs(word_pairs)
    analyzer.compute_delta_vectors()

    # Get coherence metrics
    coherence = analyzer.analyze_coherence()

    # Save vector
    library = VectorLibrary()
    metadata = {
        "model": client.model,
        "dimensions": dimensions,
        "host": client.host,
        "category": category,
        "num_pairs": len(word_pairs),
        "coherence_score": coherence["coherence_score"],
        "mean_similarity": coherence["mean_pairwise_similarity"],
    }

    library.save_vector(name, analyzer.mean_vector, word_pairs, metadata)

    print(f"\n✅ Vector '{name}' saved successfully!")
    print(f"   Coherence Score: {coherence['coherence_score']:.4f}")
    print(f"   Mean Similarity: {coherence['mean_pairwise_similarity']:.4f}")


def list_saved_vectors():
    """List all saved vectors."""
    library = VectorLibrary()
    vectors = library.list_vectors()

    if not vectors:
        print("\n📭 No saved vectors found.")
        print("   Use 'python vector_library.py save' to save a vector")
        return

    print("\n" + "=" * 80)
    print("   📚 Saved Vectors")
    print("=" * 80)

    for i, vec in enumerate(vectors, 1):
        print(f"\n{i}. {vec['name']}")
        print(f"   Created: {vec['created_at']}")
        print(f"   Pairs: {len(vec['word_pairs'])}")
        if vec["metadata"]:
            model = vec['metadata'].get('model', 'unknown')
            dims = vec['metadata'].get('dimensions', 'unknown')
            print(f"   Model: {model} ({dims} dims)")
            if "coherence_score" in vec["metadata"]:
                print(f"   Coherence: {vec['metadata']['coherence_score']:.4f}")


def apply_saved_vector():
    """Apply a saved vector to new words."""
    library = VectorLibrary()
    vectors = library.list_vectors()

    if not vectors:
        print("\n📭 No saved vectors found.")
        return

    # Show available vectors
    print("\n" + "=" * 80)
    print("   🎯 Apply Saved Vector")
    print("=" * 80)

    print("\nAvailable vectors:")
    for i, vec in enumerate(vectors, 1):
        print(f"  {i}. {vec['name']}")

    # Get choice
    try:
        choice = int(input("\nSelect vector (number): ").strip())
        if choice < 1 or choice > len(vectors):
            print("❌ Invalid choice")
            return
        vector_name = vectors[choice - 1]["name"]
    except ValueError:
        print("❌ Invalid input")
        return

    # Get words to transform
    words_input = input("\nEnter words to transform (comma-separated): ").strip()
    words = [w.strip() for w in words_input.split(",")]

    # Connect to Ollama
    try:
        client = OllamaEmbeddings()
        
        # Get dimensions
        test_emb = client.get_embedding("test")
        dimensions = test_emb.shape[0]
        
        print(f"\n📊 EMBEDDING MODEL: {client.model}")
        print(f"📏 Vector Dimensions: {dimensions}")
    except (ConnectionError, ValueError) as e:
        print(f"❌ {e}")
        return

    # Apply vector
    results = library.apply_vector(vector_name, words, client, top_k=5)

    # Display results
    print("\n" + "=" * 80)
    print(f"   Results: '{vector_name}' vector")
    print("=" * 80)

    for word in words:
        print(f"\n{word} + {vector_name} →")
        for similar_word, similarity in results[word]:
            print(f"  {similarity:.3f} : {similar_word}")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("=" * 80)
        print("   💾 Vector Library - Persist Semantic Vectors")
        print("=" * 80)
        print("\nUsage:")
        print("  python vector_library.py save      # Save a vector")
        print("  python vector_library.py list      # List saved vectors")
        print("  python vector_library.py apply     # Apply a saved vector")
        print("  python vector_library.py delete <name>  # Delete a vector")
        return

    command = sys.argv[1]

    if command == "save":
        save_from_convergence()
    elif command == "list":
        list_saved_vectors()
    elif command == "apply":
        apply_saved_vector()
    elif command == "delete":
        if len(sys.argv) < 3:
            print("Usage: python vector_library.py delete <name>")
            return
        library = VectorLibrary()
        library.delete_vector(sys.argv[2])
    else:
        print(f"Unknown command: {command}")
        print("Available commands: save, list, apply, delete")


if __name__ == "__main__":
    main()

