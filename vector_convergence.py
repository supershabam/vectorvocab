#!/usr/bin/env python3
"""
Vector Convergence Analysis

Analyze how a semantic vector converges as you add more word pair examples.
Shows how the vector stabilizes and which pairs are most representative.

Example: Computing a "masculinity" vector from man/woman, king/queen, etc.
- Shows convergence with 1, 2, 3... N pairs
- Ranks pairs by how well they align with the final average
- Identifies outliers that don't fit the pattern
"""

import sys
from typing import List, Tuple, Dict
import numpy as np

from ollama_client import OllamaEmbeddings
from utils import cosine_similarity


class VectorConvergenceAnalyzer:
    """Analyze convergence of word pair vectors with progressive analysis."""

    def __init__(self, client: OllamaEmbeddings):
        self.client = client
        self.word_pairs: List[Tuple[str, str]] = []
        self.delta_vectors: Dict[Tuple[str, str], np.ndarray] = {}
        self.mean_vector: np.ndarray = None

    def add_word_pairs(self, pairs: List[Tuple[str, str]]):
        """
        Add word pairs to analyze.

        Args:
            pairs: List of (word1, word2) tuples where word1 - word2 = delta vector
        """
        self.word_pairs = pairs

    def compute_delta_vectors(self):
        """Compute delta vectors for all word pairs."""
        print(f"\n🔍 Computing delta vectors for {len(self.word_pairs)} pairs...")

        # Get all unique words
        all_words = list(set([w for pair in self.word_pairs for w in pair]))
        print(f"   Getting embeddings for {len(all_words)} unique words...")

        # Get embeddings
        embeddings = self.client.get_embeddings(all_words)
        word_to_embedding = {word: emb for word, emb in zip(all_words, embeddings)}

        # Compute deltas
        for word1, word2 in self.word_pairs:
            delta = word_to_embedding[word1] - word_to_embedding[word2]
            self.delta_vectors[(word1, word2)] = delta

        # Compute mean vector
        self.mean_vector = np.mean(list(self.delta_vectors.values()), axis=0)

        print(f"   ✓ Computed {len(self.delta_vectors)} delta vectors")

    def analyze_progressive_convergence(self):
        """
        Analyze how the mean vector converges as we add more pairs.
        Shows stability and variance at each step.
        """
        if not self.delta_vectors:
            raise ValueError("Must compute delta vectors first")

        print("\n" + "=" * 90)
        print("📈 Progressive Convergence Analysis")
        print("=" * 90)
        print("\nShowing how the vector stabilizes as more word pairs are added...\n")

        pairs = list(self.delta_vectors.keys())
        vectors = [self.delta_vectors[p] for p in pairs]

        # Track metrics at each step
        convergence_data = []

        print(f"{'Step':<6} {'Pairs Added':<35} {'Mean Sim':<12} {'Std Dev':<12} {'Change':<12}")
        print("-" * 90)

        prev_mean = None

        for i in range(1, len(vectors) + 1):
            # Compute mean using first i vectors
            current_vectors = vectors[:i]
            current_mean = np.mean(current_vectors, axis=0)

            # Compute pairwise similarities within this subset
            sims = []
            for j in range(i):
                sim = cosine_similarity(current_vectors[j], current_mean)
                sims.append(sim)

            mean_sim = np.mean(sims)
            std_sim = np.std(sims)

            # Compute change from previous mean
            if prev_mean is not None:
                change = cosine_similarity(current_mean, prev_mean)
                change_str = f"{change:.6f}"
                stability = "✅" if change > 0.99 else "🟡" if change > 0.95 else "🔄"
            else:
                change_str = "—"
                stability = "🔄"

            pair_name = f"{pairs[i - 1][0]}-{pairs[i - 1][1]}"
            print(
                f"{i:<6} {pair_name:<35} {mean_sim:.6f}   {std_sim:.6f}   "
                f"{change_str:<12} {stability}"
            )

            convergence_data.append(
                {
                    "n_pairs": i,
                    "mean_similarity": mean_sim,
                    "std_similarity": std_sim,
                    "change": change if prev_mean is not None else None,
                }
            )

            prev_mean = current_mean

        # Summary
        print("\n" + "=" * 90)
        print("🎯 Convergence Summary")
        print("=" * 90)

        # Check if converged
        last_5_changes = [
            d["change"] for d in convergence_data[-5:] if d["change"] is not None
        ]
        if last_5_changes:
            recent_stability = np.mean(last_5_changes)
            print(f"\nRecent Stability (last 5 steps): {recent_stability:.6f}")

            if recent_stability > 0.99:
                print("✅ CONVERGED: Vector is highly stable")
                print("   → Adding more pairs won't significantly change the vector")
            elif recent_stability > 0.95:
                print("🟡 CONVERGING: Vector is mostly stable")
                print("   → Could benefit from a few more pairs")
            else:
                print("🔄 EVOLVING: Vector still changing significantly")
                print("   → Add more pairs for better stability")

        # Final coherence
        final_data = convergence_data[-1]
        print(f"\nFinal Mean Similarity: {final_data['mean_similarity']:.6f}")
        print(f"Final Std Deviation:   {final_data['std_similarity']:.6f}")

        if final_data["mean_similarity"] > 0.9:
            print("\n✨ Excellent coherence! All pairs align well with the average.")
        elif final_data["mean_similarity"] > 0.8:
            print("\n👍 Good coherence. Most pairs represent the same concept.")
        else:
            print("\n⚠️  Moderate coherence. Some pairs may represent different concepts.")

        return convergence_data

    def compute_convergence_scores(self) -> List[Tuple[Tuple[str, str], float, float]]:
        """
        Compute convergence scores for each pair.

        Returns:
            List of (pair, similarity_to_mean, variance) tuples, sorted by similarity
        """
        if not self.delta_vectors or self.mean_vector is None:
            raise ValueError("Must compute delta vectors first")

        scores = []

        for pair, delta in self.delta_vectors.items():
            # Similarity to mean vector
            sim = cosine_similarity(delta, self.mean_vector)

            # Variance (how much this vector differs from mean)
            variance = np.linalg.norm(delta - self.mean_vector)

            scores.append((pair, sim, variance))

        # Sort by similarity (highest first)
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores

    def compute_pairwise_similarities(self) -> np.ndarray:
        """
        Compute pairwise cosine similarities between all delta vectors.

        Returns:
            Matrix of pairwise similarities
        """
        n = len(self.delta_vectors)
        pairs = list(self.delta_vectors.keys())
        similarity_matrix = np.zeros((n, n))

        for i, pair_i in enumerate(pairs):
            for j, pair_j in enumerate(pairs):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = cosine_similarity(
                        self.delta_vectors[pair_i], self.delta_vectors[pair_j]
                    )
                    similarity_matrix[i, j] = sim

        return similarity_matrix

    def analyze_coherence(self) -> Dict[str, float]:
        """
        Analyze overall coherence of the vector set.

        Returns:
            Dictionary with coherence metrics
        """
        if not self.delta_vectors:
            raise ValueError("Must compute delta vectors first")

        # Compute pairwise similarities
        sim_matrix = self.compute_pairwise_similarities()

        # Get upper triangle (excluding diagonal)
        n = len(self.delta_vectors)
        upper_triangle = sim_matrix[np.triu_indices(n, k=1)]

        return {
            "mean_pairwise_similarity": float(np.mean(upper_triangle)),
            "std_pairwise_similarity": float(np.std(upper_triangle)),
            "min_pairwise_similarity": float(np.min(upper_triangle)),
            "max_pairwise_similarity": float(np.max(upper_triangle)),
            "median_pairwise_similarity": float(np.median(upper_triangle)),
            "coherence_score": float(np.mean(upper_triangle)),
        }

    def display_results(self):
        """Display analysis results in a formatted way."""
        print("\n" + "=" * 90)
        print("📊 Final Vector Analysis")
        print("=" * 90)

        # Convergence scores
        scores = self.compute_convergence_scores()

        print("\n🎯 Ranking: Similarity to Final Average Vector")
        print("-" * 90)
        print(f"{'Rank':<6} {'Word Pair':<35} {'Similarity':<15} {'Variance':<12}")
        print("-" * 90)

        for rank, (pair, sim, var) in enumerate(scores, 1):
            pair_str = f"{pair[0]} - {pair[1]}"
            indicator = (
                "🟢"
                if sim > 0.9
                else "🟡" if sim > 0.8 else "🟠" if sim > 0.7 else "🔴"
            )
            print(f"{rank:<6} {pair_str:<35} {sim:.6f} {indicator}    {var:.6f}")

        # Coherence metrics
        print("\n" + "=" * 90)
        print("📈 Overall Coherence Metrics")
        print("=" * 90)

        coherence = self.analyze_coherence()

        print(f"\nMean Pairwise Similarity:   {coherence['mean_pairwise_similarity']:.6f}")
        print(f"Std Dev:                    {coherence['std_pairwise_similarity']:.6f}")
        print(f"Min Similarity:             {coherence['min_pairwise_similarity']:.6f}")
        print(f"Max Similarity:             {coherence['max_pairwise_similarity']:.6f}")
        print(f"Median Similarity:          {coherence['median_pairwise_similarity']:.6f}")

        print(f"\n🎯 Overall Coherence Score: {coherence['coherence_score']:.6f}")

        if coherence["coherence_score"] > 0.9:
            print("   ✅ Excellent! Word pairs are highly coherent.")
        elif coherence["coherence_score"] > 0.8:
            print("   👍 Good coherence. Most pairs align well.")
        elif coherence["coherence_score"] > 0.7:
            print("   ⚠️  Moderate coherence. Some pairs may not fit the pattern.")
        else:
            print("   ❌ Low coherence. Pairs represent different semantic dimensions.")

        # Identify outliers
        print("\n" + "=" * 90)
        print("🔍 Outlier Detection")
        print("=" * 90)

        mean_sim = coherence["mean_pairwise_similarity"]
        std_sim = coherence["std_pairwise_similarity"]
        threshold = mean_sim - std_sim

        outliers = [(pair, sim) for pair, sim, _ in scores if sim < threshold]

        if outliers:
            print(f"\n⚠️  Pairs below threshold ({threshold:.4f}):")
            for pair, sim in outliers:
                print(f"   • {pair[0]:<15} - {pair[1]:<15} (similarity: {sim:.4f})")
            print("\n💡 Consider removing these pairs for a more coherent vector.")
        else:
            print("\n✅ No outliers detected. All pairs are well-aligned!")

        # Best pairs to keep
        print("\n" + "=" * 90)
        print("⭐ Top Pairs (Most Representative)")
        print("=" * 90)

        top_n = min(5, len(scores))
        print(f"\nThe top {top_n} pairs best represent this semantic dimension:\n")

        for rank, (pair, sim, _) in enumerate(scores[:top_n], 1):
            print(f"  {rank}. {pair[0]:<15} - {pair[1]:<15} (similarity: {sim:.6f})")

        print("\n💡 Use these pairs for the most accurate vector computation!")

        print("\n" + "=" * 90)


def analyze_gender_vectors():
    """Analyze convergence of gender-related word pairs."""
    try:
        client = OllamaEmbeddings()
    except (ConnectionError, ValueError) as e:
        print(f"❌ {e}")
        print("\n💡 Make sure Ollama is running:")
        print("   ollama pull nomic-embed-text")
        sys.exit(1)

    # Get dimensions
    test_emb = client.get_embedding("test")
    dimensions = test_emb.shape[0]
    
    print("=" * 90)
    print("🔬 Analyzing Gender Vector Convergence (Masculinity)")
    print("=" * 90)
    print(f"\n📊 EMBEDDING MODEL: {client.model}")
    print(f"📏 Vector Dimensions: {dimensions}")
    print(f"🌐 Host: {client.host}")

    # Define gender word pairs (masculine - feminine)
    gender_pairs = [
        ("man", "woman"),
        ("king", "queen"),
        ("boy", "girl"),
        ("father", "mother"),
        ("brother", "sister"),
        ("uncle", "aunt"),
        ("prince", "princess"),
        ("actor", "actress"),
        ("waiter", "waitress"),
        ("hero", "heroine"),
        ("sir", "madam"),
        ("gentleman", "lady"),
        ("duke", "duchess"),
        ("nephew", "niece"),
        ("husband", "wife"),
    ]

    print(f"\nAnalyzing {len(gender_pairs)} gender word pairs...")

    analyzer = VectorConvergenceAnalyzer(client)
    analyzer.add_word_pairs(gender_pairs)
    analyzer.compute_delta_vectors()

    # Show progressive convergence
    analyzer.analyze_progressive_convergence()

    # Show final analysis
    analyzer.display_results()


def analyze_custom_pairs(pairs: List[Tuple[str, str]], name: str = "Custom"):
    """Analyze convergence of custom word pairs."""
    try:
        client = OllamaEmbeddings()
    except (ConnectionError, ValueError) as e:
        print(f"❌ {e}")
        print("\n💡 Make sure Ollama is running:")
        print("   ollama pull nomic-embed-text")
        sys.exit(1)

    # Get dimensions
    test_emb = client.get_embedding("test")
    dimensions = test_emb.shape[0]
    
    print("=" * 90)
    print(f"🔬 Analyzing {name} Vector Convergence")
    print("=" * 90)
    print(f"\n📊 EMBEDDING MODEL: {client.model}")
    print(f"📏 Vector Dimensions: {dimensions}")
    print(f"🌐 Host: {client.host}")
    print(f"\nAnalyzing {len(pairs)} word pairs...")

    analyzer = VectorConvergenceAnalyzer(client)
    analyzer.add_word_pairs(pairs)
    analyzer.compute_delta_vectors()

    # Show progressive convergence
    analyzer.analyze_progressive_convergence()

    # Show final analysis
    analyzer.display_results()


def main():
    """Main function."""
    print("=" * 90)
    print("   🎯 Vector Convergence Analysis")
    print("   See how semantic vectors stabilize with more examples")
    print("=" * 90)

    if len(sys.argv) > 1:
        if sys.argv[1] == "geography":
            # Analyze geography: capital - country
            pairs = [
                ("paris", "france"),
                ("london", "england"),
                ("rome", "italy"),
                ("berlin", "germany"),
                ("madrid", "spain"),
                ("tokyo", "japan"),
                ("beijing", "china"),
                ("moscow", "russia"),
            ]
            analyze_custom_pairs(pairs, "Geography (Capital - Country)")

        elif sys.argv[1] == "size":
            # Analyze size: big - small
            pairs = [
                ("big", "small"),
                ("large", "tiny"),
                ("huge", "minuscule"),
                ("giant", "dwarf"),
                ("massive", "minute"),
                ("enormous", "microscopic"),
            ]
            analyze_custom_pairs(pairs, "Size (Big - Small)")

        elif sys.argv[1] == "animal":
            # Analyze animals: adult - young
            pairs = [
                ("dog", "puppy"),
                ("cat", "kitten"),
                ("cow", "calf"),
                ("horse", "foal"),
                ("sheep", "lamb"),
                ("lion", "cub"),
                ("bird", "chick"),
            ]
            analyze_custom_pairs(pairs, "Animals (Adult - Young)")

        else:
            print(f"Unknown category: {sys.argv[1]}")
            print("\nAvailable categories:")
            print("  gender      - Gender word pairs (default)")
            print("  geography   - Capital city to country pairs")
            print("  size        - Size comparison pairs")
            print("  animal      - Animal adult to young pairs")
            sys.exit(1)
    else:
        # Default: analyze gender vectors
        analyze_gender_vectors()


if __name__ == "__main__":
    main()
