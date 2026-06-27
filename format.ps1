# Format and lint all code with Isort, Black and Ruff

$ErrorActionPreference = "Stop"

Write-Host "================================"
Write-Host "  Code Formatting & Linting"
Write-Host "================================"
Write-Host ""

# Run Isort
Write-Host "Running Isort..." -ForegroundColor Cyan
isort .
Write-Host "Isort completed" -ForegroundColor Green

Write-Host ""

# Run Black
Write-Host "Running Black formatter..." -ForegroundColor Cyan
black .
Write-Host "Black formatting completed" -ForegroundColor Green

Write-Host ""

# Run Ruff
Write-Host "Running Ruff linter with auto-fix..." -ForegroundColor Cyan
ruff check --fix .
Write-Host "Ruff linting completed" -ForegroundColor Green

Write-Host ""
Write-Host "================================"
Write-Host "  Formatting Complete!"
Write-Host "================================"
