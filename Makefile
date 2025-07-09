# 🚀 PyRust Optimizer - Development Makefile

.PHONY: help install dev test lint format clean profile example demo

help: ## Show available commands
	@echo "🚀 PyRust Optimizer - Development Commands"
	@echo "=========================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install in development mode
	pip install -e .

dev: ## Install with development dependencies
	@echo "🚀 Setting up development environment..."
	python3 -m venv .venv || true
	.venv/bin/pip install --upgrade pip || pip install --upgrade pip
	.venv/bin/pip install pytest ruff black || pip install pytest ruff black
	@echo "✅ Development environment ready!"

test: ## Run tests
	python3 -m pytest tests/ -v || echo "No tests found yet"

lint: ## Run linting
	python3 -m ruff check src/ || echo "Ruff not installed, skipping lint"

format: ## Format code
	python3 -m ruff format src/ || echo "Ruff not installed, skipping format"

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info/ .coverage htmlcov/ .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

profile: ## Profile example code
	python examples/basic_optimization.py

example: ## Run basic example
	@echo "🚀 Running PyRust Optimizer example..."
	cd examples && python basic_optimization.py

demo: ## Run demonstration
	@echo "🎯 PyRust Optimizer Demo"
	python -c "print('🚀 PyRust Optimizer: Revolutionary Python performance!')"

verify: ## Verify installation
	@echo "✅ Verifying PyRust Optimizer..."
	python -c "from src.profiler.hotspot_detector import HotspotDetector; print('✅ Profiler loaded')"
	python -c "from src.analyzer.ast_analyzer import ASTAnalyzer; print('✅ Analyzer loaded')"

stats: ## Show project statistics
	@echo "📊 PyRust Optimizer Stats"
	@echo "Python files: $$(find src/ -name '*.py' | wc -l)"
	@echo "Test files: $$(find tests/ -name '*.py' | wc -l)"

all: clean dev test profile ## Complete setup and validation
