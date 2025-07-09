#!/bin/bash

# PyRust Optimizer Development Setup
echo "🚀 Setting up PyRust Optimizer development environment..."

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install development dependencies
pip install --upgrade pip
pip install pytest ruff black

echo "✅ Development environment setup complete!"
echo "To activate: source .venv/bin/activate"
