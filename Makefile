.PHONY: help install install-dev sync test lint format check run interactive examples convergence clean

help:  ## Show this help message
	@echo ""
	@echo "🏠 VECTORVOCAB - Local Word Vector Mathematics"
	@echo "   100% Private, 100% Local with Ollama"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ============================================
# Setup & Installation
# ============================================

install:  ## Install dependencies with uv
	uv pip install -e .

install-dev:  ## Install development dependencies
	uv pip install -e ".[dev]"

sync:  ## Sync dependencies with uv
	uv pip sync

venv:  ## Create virtual environment with uv
	uv venv

init:  ## Initialize project (create venv and install dependencies)
	uv venv
	. .venv/bin/activate && uv pip install -e ".[dev]"
	@echo ""
	@echo "✅ Project initialized!"
	@echo "💡 Run: source .venv/bin/activate"
	@echo "💡 Make sure Ollama is running: ollama pull nomic-embed-text"

# ============================================
# Main Application (Local Ollama)
# ============================================

run:  ## Run main demo (local with Ollama)
	@echo "🏠 Running local demo with Ollama..."
	uv run python vectorvocab.py

interactive:  ## Run interactive mode (local)
	@echo "🏠 Starting local interactive mode..."
	uv run python interactive.py

examples:  ## Run example analogies (local)
	@echo "🏠 Running examples locally..."
	uv run python examples.py

convergence:  ## Analyze vector convergence (gender/masculinity)
	uv run python vector_convergence.py

convergence-geography:  ## Analyze geography vector convergence (capital-country)
	uv run python vector_convergence.py geography

convergence-size:  ## Analyze size vector convergence (big-small)
	uv run python vector_convergence.py size

convergence-animal:  ## Analyze animal vector convergence (adult-young)
	uv run python vector_convergence.py animal

# ============================================
# Vector Library (Persistence)
# ============================================

vector-save:  ## Save a converged vector to library
	uv run python vector_library.py save

vector-list:  ## List all saved vectors
	uv run python vector_library.py list

vector-apply:  ## Apply a saved vector to new words
	uv run python vector_library.py apply

# ============================================
# Ollama Management
# ============================================

test:  ## Test Ollama connection
	uv run python ollama_client.py

list-models:  ## List available Ollama models
	uv run python ollama_client.py list

pull-model:  ## Pull an Ollama model (usage: make pull-model MODEL=nomic-embed-text)
	@if [ -z "$(MODEL)" ]; then \
		echo "Usage: make pull-model MODEL=nomic-embed-text"; \
		exit 1; \
	fi
	ollama pull $(MODEL)

# ============================================
# Code Quality
# ============================================

lint:  ## Run ruff linter
	uv run ruff check .

format:  ## Format code with ruff
	uv run ruff format .

check:  ## Run linter and formatter (check only)
	uv run ruff check .
	uv run ruff format --check .

fix:  ## Fix linting issues and format code
	uv run ruff check --fix .
	uv run ruff format .

# ============================================
# Cleanup
# ============================================

clean:  ## Clean cache files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	@echo "✅ Cleaned up cache files"
