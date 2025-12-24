#!/usr/bin/env python3
"""
Ollama Client for Local Word Embeddings

Run embedding models 100% locally and privately using Ollama.
No API keys, no cloud, no data leaving your machine!
"""

import requests
import numpy as np
from typing import List


class OllamaEmbeddings:
    """Client for Ollama local embeddings."""
    
    AVAILABLE_MODELS = {
        'nomic-embed-text': {
            'name': 'nomic-embed-text',
            'description': 'High-performing open embedding model with large token context',
            'params': '48M',
            'dimensions': 768,
            'context': 8192,
        },
        'mxbai-embed-large': {
            'name': 'mxbai-embed-large',
            'description': 'State-of-the-art large embedding model from mixedbread.ai',
            'params': '335M',
            'dimensions': 1024,
            'context': 512,
        },
        'bge-m3': {
            'name': 'bge-m3',
            'description': 'Multi-Functionality, Multi-Linguality, Multi-Granularity',
            'params': '567M',
            'dimensions': 1024,
            'context': 8192,
        },
        'all-minilm': {
            'name': 'all-minilm',
            'description': 'Lightweight and fast, great for development',
            'params': '22M-33M',
            'dimensions': 384,
            'context': 256,
        },
        'snowflake-arctic-embed': {
            'name': 'snowflake-arctic-embed',
            'description': 'Optimized for performance',
            'params': '22M-335M',
            'dimensions': 1024,
            'context': 512,
        },
        'bge-large': {
            'name': 'bge-large',
            'description': 'Large embedding model from BAAI',
            'params': '335M',
            'dimensions': 1024,
            'context': 512,
        },
    }
    
    def __init__(self, model: str = 'nomic-embed-text', host: str = 'http://localhost:11434'):
        """
        Initialize Ollama embeddings client.
        
        Args:
            model: Name of the Ollama model to use
            host: Ollama API endpoint (default: http://localhost:11434)
        """
        self.model = model
        self.host = host.rstrip('/')
        self.api_url = f"{self.host}/api/embeddings"
        
        # Verify Ollama is running
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=2)
            if response.status_code != 200:
                raise ConnectionError("Cannot connect to Ollama")
        except requests.exceptions.RequestException:
            raise ConnectionError(
                f"Cannot connect to Ollama at {host}\n"
                f"Make sure Ollama is installed and running:\n"
                f"  1. Install from https://ollama.com/download\n"
                f"  2. Run: ollama pull {model}\n"
                f"  3. Ollama should start automatically"
            )
        
        # Check if model is available
        self._ensure_model_available()
    
    def _ensure_model_available(self):
        """Check if the model is pulled and available."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [m['name'].split(':')[0] for m in data.get('models', [])]
                
                if self.model not in models:
                    print(f"\n⚠️  Model '{self.model}' not found locally")
                    print(f"   To download it, run:")
                    print(f"   ollama pull {self.model}")
                    raise ValueError(f"Model '{self.model}' not available")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Error checking models: {e}")
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            numpy array of shape (len(texts), dimensions)
        """
        embeddings = []
        
        for text in texts:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": text
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"API Error: {response.status_code} - {response.text}")
            
            result = response.json()
            
            if "embedding" in result:
                embeddings.append(result["embedding"])
            else:
                raise Exception(f"Unexpected response format: {result}")
        
        return np.array(embeddings)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
        return self.get_embeddings([text])[0]
    
    @classmethod
    def list_available_models(cls):
        """Print available embedding models."""
        print("\n" + "="*70)
        print("Available Ollama Embedding Models")
        print("="*70)
        
        for model_id, info in cls.AVAILABLE_MODELS.items():
            print(f"\n📦 {info['name']}")
            print(f"   {info['description']}")
            print(f"   Parameters: {info['params']}")
            print(f"   Dimensions: {info['dimensions']}")
            print(f"   Context: {info['context']} tokens")
            print(f"\n   To use: ollama pull {info['name']}")
        
        print("\n" + "="*70)
        print(f"Full list: https://ollama.com/search?c=embedding")
        print("="*70)


def test_ollama():
    """Test Ollama connection and embeddings."""
    print("="*70)
    print("    Testing Ollama Embeddings")
    print("="*70)
    
    try:
        # Try with nomic-embed-text (popular choice)
        print("\n🔍 Connecting to Ollama...")
        client = OllamaEmbeddings(model='nomic-embed-text')
        
        print(f"✅ Connected! Using model: {client.model}")
        
        # Test with simple words
        print("\n📊 Testing embeddings...")
        words = ["king", "queen", "man", "woman"]
        embeddings = client.get_embeddings(words)
        
        print(f"✅ Success! Generated {len(embeddings)} embeddings")
        print(f"   Dimensions: {embeddings.shape[1]}")
        print(f"   First 10 values of 'king': {embeddings[0][:10]}")
        
        # Test cosine similarity
        from utils import cosine_similarity
        sim = cosine_similarity(embeddings[0], embeddings[1])  # king vs queen
        print(f"\n🎯 Similarity between 'king' and 'queen': {sim:.4f}")
        
        print("\n" + "="*70)
        print("✨ Ollama is working perfectly!")
        print("="*70)
        
    except ConnectionError as e:
        print(f"\n❌ Connection Error: {e}")
        print("\n💡 To get started with Ollama:")
        print("   1. Download from: https://ollama.com/download")
        print("   2. Install and start Ollama")
        print("   3. Pull a model: ollama pull nomic-embed-text")
        print("   4. Run this test again")
    except ValueError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'list':
        OllamaEmbeddings.list_available_models()
    else:
        test_ollama()

