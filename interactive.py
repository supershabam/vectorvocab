#!/usr/bin/env python3
"""
Interactive Word Vector Explorer (Ollama Version)

Allows you to experiment with word vector arithmetic in an interactive session
using local Ollama models - 100% private and offline!
"""

import sys
import numpy as np
from ollama_client import OllamaEmbeddings
from utils import cosine_similarity, find_closest_words


class VectorExplorerOllama:
    """Interactive word vector exploration tool using Ollama."""
    
    def __init__(self, model='nomic-embed-text'):
        try:
            self.client = OllamaEmbeddings(model=model)
            self.embedding_cache = {}
            self.model = model
        except (ConnectionError, ValueError) as e:
            print(f"\n❌ Error: {e}")
            sys.exit(1)
    
    def get_embedding(self, word: str) -> np.ndarray:
        """Get embedding for a word, using cache if available."""
        if word not in self.embedding_cache:
            print(f"  [Fetching embedding for '{word}']")
            self.embedding_cache[word] = self.client.get_embedding(word)
        return self.embedding_cache[word]
    
    def compute_analogy(self, word_a: str, word_b: str, word_c: str, num_results: int = 5):
        """
        Solve word analogy: word_a - word_b + word_c = ?
        Example: king - man + woman = queen
        """
        print(f"\n🔍 Computing: {word_a} - {word_b} + {word_c} = ?\n")
        
        # Get embeddings
        emb_a = self.get_embedding(word_a)
        emb_b = self.get_embedding(word_b)
        emb_c = self.get_embedding(word_c)
        
        # Compute result vector
        result_vector = emb_a - emb_b + emb_c
        
        # Get common words to search through
        common_words = self.get_common_words()
        
        # Find closest matches
        print(f"🎯 Closest matches:")
        closest = find_closest_words(
            result_vector,
            self.embedding_cache,
            top_k=num_results,
            exclude=[word_a, word_b, word_c]
        )
        
        for i, (word, similarity) in enumerate(closest, 1):
            print(f"  {i}. {word:15} (similarity: {similarity:.4f})")
    
    def get_common_words(self) -> list:
        """Get a list of common words to search through."""
        common_words = [
            # People
            "man", "woman", "king", "queen", "boy", "girl", "father", "mother",
            "brother", "sister", "uncle", "aunt", "prince", "princess",
            "actor", "actress", "waiter", "waitress", "hero", "heroine",
            "son", "daughter", "husband", "wife", "gentleman", "lady",
            
            # Professions
            "doctor", "nurse", "teacher", "professor", "engineer", "scientist",
            "artist", "musician", "chef", "cook", "dancer", "singer",
            "writer", "author", "poet", "painter", "programmer", "developer",
            
            # Animals
            "cat", "dog", "lion", "tiger", "bear", "wolf", "fox",
            "bird", "eagle", "owl", "hawk", "duck", "goose",
            
            # Places
            "paris", "france", "london", "england", "rome", "italy",
            "berlin", "germany", "tokyo", "japan", "beijing", "china",
            
            # Things
            "pizza", "pasta", "bread", "cake", "coffee", "tea",
            "car", "truck", "bike", "plane", "train", "boat",
            
            # Abstract
            "love", "hate", "happy", "sad", "good", "bad", "big", "small",
            "hot", "cold", "fast", "slow", "strong", "weak",
        ]
        
        # Ensure all words are in cache
        words_to_fetch = [w for w in common_words if w not in self.embedding_cache]
        if words_to_fetch:
            print(f"  [Loading {len(words_to_fetch)} common words...]")
            embeddings = self.client.get_embeddings(words_to_fetch)
            for word, emb in zip(words_to_fetch, embeddings):
                self.embedding_cache[word] = emb
        
        return common_words
    
    def compare_words(self, word1: str, word2: str):
        """Compare two words and show their similarity."""
        print(f"\n📊 Comparing '{word1}' and '{word2}'\n")
        
        emb1 = self.get_embedding(word1)
        emb2 = self.get_embedding(word2)
        
        similarity = cosine_similarity(emb1, emb2)
        
        print(f"  Cosine Similarity: {similarity:.4f}")
        
        if similarity > 0.8:
            print("  → Very similar! 🎯")
        elif similarity > 0.6:
            print("  → Somewhat similar 👍")
        elif similarity > 0.4:
            print("  → Weakly similar 🤔")
        else:
            print("  → Not very similar 🚫")
    
    def find_similar(self, word: str, num_results: int = 10):
        """Find words similar to the given word."""
        print(f"\n🔎 Finding words similar to '{word}'\n")
        
        emb = self.get_embedding(word)
        
        # Load common words
        self.get_common_words()
        
        closest = find_closest_words(
            emb,
            self.embedding_cache,
            top_k=num_results + 1,  # +1 because we'll exclude the word itself
            exclude=[word]
        )
        
        print(f"📋 Most similar words:")
        for i, (similar_word, similarity) in enumerate(closest, 1):
            print(f"  {i:2}. {similar_word:15} (similarity: {similarity:.4f})")
    
    def run_interactive(self):
        """Run interactive mode."""
        # Get dimensions
        test_emb = self.client.get_embedding("test")
        dimensions = test_emb.shape[0]
        
        print("\n" + "="*60)
        print("    🎨 Interactive Word Vector Explorer (Ollama)")
        print("    100% Local & Private!")
        print("="*60)
        print(f"\n📊 EMBEDDING MODEL: {self.model}")
        print(f"📏 Vector Dimensions: {dimensions}")
        print(f"🌐 Host: {self.client.host}")
        print("\nCommands:")
        print("  analogy <word1> <word2> <word3>  - Solve: word1 - word2 + word3 = ?")
        print("  compare <word1> <word2>           - Compare similarity of two words")
        print("  similar <word>                    - Find similar words")
        print("  quit / exit                       - Exit the program")
        print("\nExamples:")
        print("  analogy king man woman")
        print("  compare doctor nurse")
        print("  similar pizza")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                parts = user_input.lower().split()
                command = parts[0]
                
                if command in ["quit", "exit", "q"]:
                    print("\n👋 Goodbye!")
                    break
                
                elif command == "analogy" and len(parts) == 4:
                    self.compute_analogy(parts[1], parts[2], parts[3])
                
                elif command == "compare" and len(parts) == 3:
                    self.compare_words(parts[1], parts[2])
                
                elif command == "similar" and len(parts) == 2:
                    self.find_similar(parts[1])
                
                else:
                    print("❌ Invalid command. Type a valid command or 'quit' to exit.")
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main function."""
    model = "nomic-embed-text"  # Default
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        model = sys.argv[1]
    
    explorer = VectorExplorerOllama(model=model)
    explorer.run_interactive()


if __name__ == "__main__":
    main()

