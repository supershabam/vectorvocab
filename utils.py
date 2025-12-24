"""
Shared utility functions for word vector operations.
"""

import numpy as np
from typing import Dict, List, Tuple


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def find_closest_words(
    target_vector: np.ndarray,
    word_embeddings: Dict[str, np.ndarray],
    top_k: int = 5,
    exclude: List[str] = None,
) -> List[Tuple[str, float]]:
    """
    Find the closest words to a target vector.

    Args:
        target_vector: The vector to compare against
        word_embeddings: Dictionary mapping words to their embeddings
        top_k: Number of results to return
        exclude: List of words to exclude from results

    Returns:
        List of (word, similarity_score) tuples
    """
    exclude = exclude or []
    similarities = []

    for word, embedding in word_embeddings.items():
        if word not in exclude:
            sim = cosine_similarity(target_vector, embedding)
            similarities.append((word, sim))

    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]

