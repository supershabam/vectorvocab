#!/usr/bin/env python3
"""
Fun Examples of Word Vector Analogies

This script demonstrates various interesting semantic relationships
discovered through word vector arithmetic using local Ollama models.
"""

import sys
import numpy as np
from ollama_client import OllamaEmbeddings
from utils import find_closest_words


def run_analogy(client: OllamaEmbeddings, word_a: str, word_b: str, word_c: str, 
                context_words: list = None):
    """
    Compute and display word analogy: word_a - word_b + word_c = ?
    """
    print(f"\n📝 {word_a} - {word_b} + {word_c} = ?")
    
    # Default context words if none provided
    if context_words is None:
        context_words = [word_a, word_b, word_c]
    
    # Get embeddings for all words
    all_words = list(set([word_a, word_b, word_c] + context_words))
    embeddings = client.get_embeddings(all_words)
    word_to_embedding = {word: emb for word, emb in zip(all_words, embeddings)}
    
    # Compute analogy
    result = (word_to_embedding[word_a] - 
              word_to_embedding[word_b] + 
              word_to_embedding[word_c])
    
    # Find closest words
    closest = find_closest_words(result, word_to_embedding, top_k=3, 
                                 exclude=[word_a, word_b, word_c])
    
    print("   →", ", ".join([f"{word} ({sim:.3f})" for word, sim in closest]))


def main():
    """Run various analogy examples."""
    # Get model from command line or use default
    model = sys.argv[1] if len(sys.argv) > 1 else "nomic-embed-text"
    
    print("="*70)
    print("    🎯 Word Vector Analogy Examples")
    print("    100% Local with Ollama")
    print("="*70)
    
    try:
        client = OllamaEmbeddings(model=model)
        
        # Get dimensions
        test_emb = client.get_embedding("test")
        dimensions = test_emb.shape[0]
        
        print(f"\n📊 EMBEDDING MODEL: {model}")
        print(f"📏 Vector Dimensions: {dimensions}\n")
        
        # ============================================
        # Gender Analogies
        # ============================================
        print("\n" + "─"*70)
        print("👫 Gender Relationships")
        print("─"*70)
        
        gender_words = [
            "man", "woman", "king", "queen", "prince", "princess",
            "boy", "girl", "father", "mother", "brother", "sister",
            "actor", "actress", "waiter", "waitress", "uncle", "aunt",
            "sir", "madam", "lord", "lady", "gentleman", "gentlewoman"
        ]
        
        run_analogy(client, "king", "man", "woman", gender_words)
        run_analogy(client, "prince", "boy", "girl", gender_words)
        run_analogy(client, "actor", "man", "woman", gender_words)
        run_analogy(client, "waiter", "man", "woman", gender_words)
        
        # ============================================
        # Geographic Analogies
        # ============================================
        print("\n" + "─"*70)
        print("🌍 Geography: Capital Cities")
        print("─"*70)
        
        geo_words = [
            "paris", "france", "london", "england", "rome", "italy",
            "berlin", "germany", "madrid", "spain", "tokyo", "japan",
            "beijing", "china", "moscow", "russia", "washington", "america",
            "canada", "ottawa", "australia", "canberra", "brazil", "brasilia"
        ]
        
        run_analogy(client, "paris", "france", "italy", geo_words)
        run_analogy(client, "london", "england", "spain", geo_words)
        run_analogy(client, "tokyo", "japan", "china", geo_words)
        run_analogy(client, "berlin", "germany", "russia", geo_words)
        
        # ============================================
        # Verb Tenses
        # ============================================
        print("\n" + "─"*70)
        print("⏰ Verb Tenses: Present to Past")
        print("─"*70)
        
        verb_words = [
            "walk", "walked", "run", "ran", "swim", "swam",
            "write", "wrote", "think", "thought", "buy", "bought",
            "eat", "ate", "drink", "drank", "sing", "sang",
            "do", "did", "go", "went", "see", "saw"
        ]
        
        run_analogy(client, "walking", "walk", "swim", verb_words)
        run_analogy(client, "ran", "run", "swim", verb_words)
        run_analogy(client, "wrote", "write", "think", verb_words)
        run_analogy(client, "bought", "buy", "sell", verb_words + ["sell", "sold"])
        
        # ============================================
        # Size Relationships
        # ============================================
        print("\n" + "─"*70)
        print("📏 Size: Big to Small")
        print("─"*70)
        
        size_words = [
            "big", "small", "large", "tiny", "huge", "minuscule",
            "giant", "dwarf", "enormous", "microscopic",
            "ocean", "puddle", "mountain", "hill", "tree", "shrub"
        ]
        
        run_analogy(client, "big", "small", "ocean", size_words)
        run_analogy(client, "huge", "tiny", "mountain", size_words)
        run_analogy(client, "giant", "dwarf", "tree", size_words)
        
        # ============================================
        # Animal Relationships
        # ============================================
        print("\n" + "─"*70)
        print("🐾 Animal: Adult to Young")
        print("─"*70)
        
        animal_words = [
            "dog", "puppy", "cat", "kitten", "cow", "calf",
            "horse", "foal", "sheep", "lamb", "lion", "cub",
            "bird", "chick", "duck", "duckling", "swan", "cygnet"
        ]
        
        run_analogy(client, "dog", "puppy", "cat", animal_words)
        run_analogy(client, "cow", "calf", "horse", animal_words)
        run_analogy(client, "lion", "cub", "bird", animal_words)
        
        # ============================================
        # Occupation Expertise
        # ============================================
        print("\n" + "─"*70)
        print("👨‍⚕️ Occupations and Expertise")
        print("─"*70)
        
        occupation_words = [
            "doctor", "medicine", "chef", "cooking", "teacher", "education",
            "lawyer", "law", "engineer", "engineering", "artist", "art",
            "musician", "music", "writer", "writing", "scientist", "science"
        ]
        
        run_analogy(client, "doctor", "medicine", "cooking", occupation_words)
        run_analogy(client, "teacher", "education", "law", occupation_words)
        run_analogy(client, "artist", "art", "music", occupation_words)
        
        # ============================================
        # Opposites
        # ============================================
        print("\n" + "─"*70)
        print("🔄 Opposites")
        print("─"*70)
        
        opposite_words = [
            "good", "bad", "hot", "cold", "fast", "slow",
            "happy", "sad", "dark", "light", "up", "down",
            "right", "wrong", "love", "hate", "peace", "war"
        ]
        
        run_analogy(client, "good", "bad", "hot", opposite_words)
        run_analogy(client, "happy", "sad", "fast", opposite_words)
        run_analogy(client, "love", "hate", "peace", opposite_words)
        
        # ============================================
        # Fun Experiments
        # ============================================
        print("\n" + "─"*70)
        print("🎨 Creative Experiments")
        print("─"*70)
        
        fun_words = [
            "coffee", "tea", "pizza", "pasta", "hamburger", "hotdog",
            "breakfast", "lunch", "dinner", "morning", "evening", "night",
            "summer", "winter", "beach", "snow"
        ]
        
        run_analogy(client, "coffee", "morning", "evening", fun_words)
        run_analogy(client, "pizza", "italy", "japan", fun_words + geo_words)
        run_analogy(client, "beach", "summer", "winter", fun_words)
        
        print("\n" + "="*70)
        print("✨ All examples complete!")
        print("="*70)
        print("\n💡 Try your own analogies with: make interactive")
        print("   Or directly: uv run python interactive.py")
        
    except (ConnectionError, ValueError) as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure Ollama is running and you have an embedding model installed:")
        print("   ollama pull nomic-embed-text")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

