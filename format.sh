#!/bin/bash
# Format and lint all code with Isort, Black and Ruff

set -e

echo "================================"
echo "  Code Formatting & Linting"
echo "================================"
echo ""

# Run Isort
echo "🎨 Running Isort..."
isort .
echo "✅ Isort completed"

echo ""

# Run Black
echo "🎨 Running Black formatter..."
black .
echo "✅ Black formatting completed"

echo ""

# Run Ruff
echo "🔍 Running Ruff linter with auto-fix..."
ruff check --fix .
echo "✅ Ruff linting completed"

echo ""
echo "================================"
echo "  Formatting Complete!"
echo "================================"
