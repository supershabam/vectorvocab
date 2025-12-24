# VectorVocab - Local Word Vector Mathematics

**100% Private. 100% Local. 100% Free.**

Explore word embeddings and vector arithmetic using **Ollama** for completely local, private semantic analysis. Discover relationships like `king - man + woman = queen` entirely on your machine!

## 🏠 Why Local?

- 🔒 **Private**: Your data never leaves your machine
- ⚡ **Fast**: Near-zero latency with local processing
- 💰 **Free**: No API costs, ever
- 🌐 **Offline**: Works without internet (after model download)
- 🎯 **Full Control**: Choose any embedding model you want

## What This Does

1. **Generate Embeddings**: Uses local Ollama models to create vector representations of words
2. **Compute Semantic Vectors**: Calculates relationship vectors (e.g., "masculinity") by analyzing word pairs
3. **Apply to New Words**: Adds computed vectors to discover semantic relationships
4. **Analyze Convergence** 🆕: Rank how well different word pairs converge onto similar vectors
5. **Track Models** 🆕: All scripts log which embedding model is used for reproducibility

## Quick Start

### 1. Install Prerequisites

**Install [Ollama](https://ollama.com/download)**:
```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Or download from: https://ollama.com/download
```

**Install [uv](https://github.com/astral-sh/uv)** (ultra-fast Python package manager):
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Setup Project

```bash
git clone <your-repo-url>
cd vectorvocab

# Initialize project (creates venv and installs deps)
make init

# Download an embedding model (274 MB)
ollama pull nomic-embed-text
```

### 3. Run!

```bash
# Main demo
make run

# Interactive exploration
make interactive

# Example analogies
make examples

# Vector convergence analysis
make convergence
```

## Usage Examples

### Main Demo
```bash
make run
# Or: uv run python vectorvocab.py
```

Demonstrates:
- Classic analogy: `king - man + woman = queen`
- Computing gender vectors from multiple word pairs
- Applying transformations to test words (pizza, doctor, nurse, etc.)

### Vector Persistence
```bash
# Save a converged vector for reuse
make vector-save

# List saved vectors
make vector-list

# Apply saved vector to new words
make vector-apply
```

Save computed vectors and reuse them without recomputation!

### Interactive Mode
```bash
make interactive
# Or: uv run python interactive.py
```

Explore word relationships interactively:
```
> analogy king man woman
  → queen (0.789)

> compare doctor nurse
  → Similarity: 0.614

> similar pizza
  → italy (0.624), pasta (0.589), rome (0.557)
```

### Example Analogies
```bash
make examples
# Or: uv run python examples.py
```

Runs 50+ pre-configured analogies across categories:
- Gender relationships
- Geographic capitals
- Verb tenses
- Size relationships
- Animal families
- Occupations
- Opposites

### Vector Convergence Analysis
```bash
make convergence
# Or: uv run python vector_convergence.py
```

Analyzes how consistently different word pairs represent a semantic relationship.

### Save and Reuse Vectors

After analyzing convergence, save your computed vectors for later use:

```bash
# 1. Analyze and save a vector
make convergence                # Analyze gender vector
make vector-save                # Save it to library

# 2. Later, reuse the saved vector
make vector-list                # See available vectors
make vector-apply               # Apply to new words
```

**Example workflow:**
1. Compute "masculinity" vector from 15 gender pairs
2. Save it to `.vectors/masculinity.json`
3. Later, apply it to any words instantly: `pizza → man, father, king`

No need to recompute embeddings - just load and apply!

## Available Models

Popular Ollama embedding models (sorted by dimensions):

| Model | Dimensions | Size | Best For |
|-------|------------|------|----------|
| **bge-m3** | **1024** | 2.2 GB | Multilingual (highest dims) |
| **bge-large-en-v1.5** | **1024** | 1.3 GB | High quality English |
| **mxbai-embed-large** | **1024** | 669 MB | Balanced quality/size |
| **nomic-embed-text** ⭐ | 768 | 274 MB | General use (Recommended) |
| **bge-base-en-v1.5** | 768 | 438 MB | Good quality |
| **all-minilm** | 384 | 46 MB | Speed & efficiency |

```bash
# Download a model
ollama pull bge-m3              # Highest dimensions (1024)
ollama pull nomic-embed-text    # Recommended default (768)

# List installed models
ollama list

# Use a different model
uv run python vectorvocab.py bge-m3
```

**Want the highest dimensions?** Use `bge-m3`, `bge-large-en-v1.5`, or `mxbai-embed-large` (all 1024 dims)

See the [full model list](https://ollama.com/search?c=embedding) on Ollama's website.

## How It Works

Word embeddings capture semantic meaning in high-dimensional vectors. Similar words have similar vectors, and relationships between words can be expressed as vector arithmetic:

- `king - man + woman ≈ queen`
- `paris - france + italy ≈ rome`
- `walking - walk + swim ≈ swimming`

The project computes relationship vectors (like gender, geography, size) and applies them to discover semantic connections.

## Example Output

```
============================================================
    VECTORVOCAB - Word Vector Mathematics
    Powered by Ollama (Local Models)
============================================================

🔍 Connecting to Ollama...
✓ Connected to Ollama
  Model: nomic-embed-text
  Host: http://localhost:11434

============================================================
Classic Word Analogy: king - man + woman = ?
============================================================

Closest words to the result:
  0.7886 : queen
  0.6553 : princess
  0.6410 : monarch

============================================================
Computing masculinity vector from word pairs...
============================================================

Word pair differences:
  man          - woman        = vector (norm: 18.196)
  king         - queen        = vector (norm: 16.817)
  boy          - girl         = vector (norm: 20.485)
  ...

============================================================
Applying masculinity vector to words...
============================================================

pizza        + masculinity_vector →
    0.537 : man
    0.505 : father
    0.502 : king
```

## Makefile Commands

```bash
make help                 # Show all available commands

# Main application (local)
make run                  # Run main demo
make interactive          # Interactive exploration
make examples             # Run example analogies
make convergence          # Analyze vector convergence

# Vector persistence (NEW!)
make vector-save          # Save a converged vector
make vector-list          # List saved vectors
make vector-apply         # Apply saved vector to words

# Ollama management
make test                 # Test Ollama connection
make list-models          # List installed models
make pull-model MODEL=... # Pull a new model

# Code quality
make lint                 # Run linter
make format               # Format code
make check                # Check code quality
make fix                  # Auto-fix issues

# Setup
make init                 # Initialize project
make install              # Install dependencies
make clean                # Clean cache files
```

## 🛠️ Development

This project uses modern Python tools:

- **[uv](https://github.com/astral-sh/uv)** - Ultra-fast package manager (10-100x faster than pip)
- **[ruff](https://github.com/astral-sh/ruff)** - Ultra-fast linter & formatter (10-100x faster than flake8/black)

## Project Structure

```
vectorvocab/
├── vectorvocab.py            # Main demo
├── interactive.py            # Interactive explorer
├── examples.py               # Pre-configured examples
├── vector_convergence.py     # Convergence analysis
├── ollama_client.py          # Ollama API client
├── utils.py                  # Shared utilities
├── vector_library.py         # Vector persistence
├── Makefile                  # Development automation
└── pyproject.toml            # Python project config
```

## Resources

### Ollama
- [Ollama Website](https://ollama.com)
- [Embedding Models](https://ollama.com/search?c=embedding)
- [Ollama GitHub](https://github.com/ollama/ollama)
- [Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)

### Word Embeddings
- [Word2Vec Paper](https://arxiv.org/abs/1301.3781)
- [Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/)
- [BGE Model on Hugging Face](https://huggingface.co/BAAI/bge-base-en-v1.5)

### Modern Python Tools
- [uv Documentation](https://github.com/astral-sh/uv)
- [ruff Documentation](https://github.com/astral-sh/ruff)

## License

MIT

## Contributing

Contributions welcome!
