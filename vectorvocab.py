#!/usr/bin/env python3
"""
VectorVocab - Word Vector Mathematics using Local Ollama Models

Explores semantic relationships through vector arithmetic:
- king - man + woman = queen
- paris - france + italy = rome

100% local, 100% private, 100% free!
"""

import sys
import numpy as np
from ollama_client import OllamaEmbeddings
from utils import cosine_similarity, find_closest_words


def compute_gender_vector(client: OllamaEmbeddings) -> tuple:
    """
    Compute a "masculinity" vector by averaging differences across multiple word pairs.
    """
    print("\n" + "="*60)
    print("Computing masculinity vector from word pairs...")
    print("="*60)
    
    # Define word pairs where first is masculine, second is feminine
    word_pairs = [
        ("man", "woman"),
        ("king", "queen"),
        ("boy", "girl"),
        ("father", "mother"),
        ("brother", "sister"),
        ("uncle", "aunt"),
        ("prince", "princess"),
        ("actor", "actress"),
    ]
    
    # Get all unique words
    all_words = list(set([w for pair in word_pairs for w in pair]))
    
    print(f"\nGetting embeddings for {len(all_words)} words...")
    embeddings = client.get_embeddings(all_words)
    
    # Create word -> embedding mapping
    word_to_embedding = {word: emb for word, emb in zip(all_words, embeddings)}
    
    # Compute difference vectors
    gender_vectors = []
    print("\nWord pair differences:")
    for masc, fem in word_pairs:
        diff = word_to_embedding[masc] - word_to_embedding[fem]
        gender_vectors.append(diff)
        print(f"  {masc:12} - {fem:12} = vector (norm: {np.linalg.norm(diff):.3f})")
    
    # Average the difference vectors
    avg_gender_vector = np.mean(gender_vectors, axis=0)
    print(f"\nAveraged gender vector (norm: {np.linalg.norm(avg_gender_vector):.3f})")
    
    return avg_gender_vector, word_to_embedding


def explore_word_transformations(client: OllamaEmbeddings,
                                 gender_vector: np.ndarray,
                                 base_embeddings: dict):
    """
    Apply the gender vector to various words and see what we get.
    """
    print("\n" + "="*60)
    print("Applying masculinity vector to words...")
    print("="*60)
    
    # Words to transform
    test_words = [
        "pizza",
        "doctor",
        "nurse",
        "teacher",
        "chef",
        "cat",
        "dog",
    ]
    
    # Get embeddings for test words
    print(f"\nGetting embeddings for {len(test_words)} test words...")
    test_embeddings = client.get_embeddings(test_words)
    test_word_to_embedding = {word: emb for word, emb in zip(test_words, test_embeddings)}
    
    # Merge with base embeddings for comparison
    all_embeddings = {**base_embeddings, **test_word_to_embedding}
    
    print("\nTransformations:")
    print("-" * 60)
    
    for word in test_words:
        # Apply the gender vector
        transformed = test_word_to_embedding[word] + gender_vector
        
        # Find closest words to the transformed vector
        closest = find_closest_words(transformed, all_embeddings, top_k=3, exclude=[word])
        
        print(f"\n{word:12} + masculinity_vector →")
        for similar_word, similarity in closest:
            print(f"    {similarity:.3f} : {similar_word}")


def demonstrate_classic_analogy(client: OllamaEmbeddings):
    """
    Demonstrate the classic word analogy: king - man + woman ≈ queen
    """
    print("\n" + "="*60)
    print("Classic Word Analogy: king - man + woman = ?")
    print("="*60)
    
    words = ["king", "man", "woman", "queen", "prince", "princess", "monarch", "ruler"]
    
    print(f"\nGetting embeddings for {len(words)} words...")
    embeddings = client.get_embeddings(words)
    word_to_embedding = {word: emb for word, emb in zip(words, embeddings)}
    
    # Compute king - man + woman
    result_vector = (word_to_embedding["king"] - 
                    word_to_embedding["man"] + 
                    word_to_embedding["woman"])
    
    print("\nComputing: king - man + woman")
    print("\nClosest words to the result:")
    
    closest = find_closest_words(result_vector, word_to_embedding, top_k=5, 
                                 exclude=["king", "man", "woman"])
    
    for word, similarity in closest:
        print(f"  {similarity:.4f} : {word}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("    VECTORVOCAB - Word Vector Mathematics")
    print("    Powered by Ollama (Local Models)")
    print("=" * 60)
    
    # Parse command line arguments
    model = "nomic-embed-text"  # Default model
    if len(sys.argv) > 1:
        model = sys.argv[1]
    
    try:
        # Initialize client
        print(f"\n🔍 Connecting to Ollama...")
        client = OllamaEmbeddings(model=model)
        
        # Get dimensions
        test_emb = client.get_embedding("test")
        dimensions = test_emb.shape[0]
        
        print(f"✓ Connected to Ollama")
        print(f"📊 EMBEDDING MODEL: {model}")
        print(f"📏 Vector Dimensions: {dimensions}")
        print(f"🌐 Host: {client.host}")
        
        # Demonstrate classic analogy
        demonstrate_classic_analogy(client)
        
        # Compute gender vector and apply to words
        gender_vector, base_embeddings = compute_gender_vector(client)
        explore_word_transformations(client, gender_vector, base_embeddings)
        
        print("\n" + "="*60)
        print("✓ Analysis complete!")
        print("="*60)
        print("\n💡 Tips:")
        print("  - Your data never leaves your machine")
        print("  - Try different models: python vectorvocab.py bge-m3")
        print("  - List models: python ollama_client.py list")
        
    except ConnectionError as e:
        print(f"\n❌ Connection Error: {e}")
        print("\n📥 To get started with Ollama:")
        print("  1. Download: https://ollama.com/download")
        print("  2. Install and start Ollama")
        print("  3. Pull a model: ollama pull nomic-embed-text")
        print("  4. Run this script again")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

