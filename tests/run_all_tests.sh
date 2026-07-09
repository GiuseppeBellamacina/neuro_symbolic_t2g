#!/bin/bash
# ============================================================================
# Batch Test Runner — Neuro-Symbolic T2G
#
# All tests are now proper pytest tests (with fixtures and assert).
# Run with:
#   bash tests/run_all_tests.sh             # Tutti i test
#   bash tests/run_all_tests.sh --skip-data # Salta test data (online)
#   bash tests/run_all_tests.sh --verbose   # Output dettagliato
# ============================================================================

set -e

cd "$(dirname "$0")/.."

SKIP_DATA=0
VERBOSE=0

for arg in "$@"; do
    case "$arg" in
        --skip-data) SKIP_DATA=1 ;;
        --verbose)   VERBOSE=1 ;;
    esac
done

# Build pytest args
PYTEST_ARGS=""

if [ "$VERBOSE" -eq 1 ]; then
    PYTEST_ARGS="$PYTEST_ARGS -v"
fi

if [ "$SKIP_DATA" -eq 1 ]; then
    # Skip data and integration tests (require dataset download)
    PYTEST_ARGS="$PYTEST_ARGS --ignore=tests/test_data.py --ignore=tests/test_integration.py"
fi

echo "========================================================"
echo "  Neuro-Symbolic T2G — Test Suite (pytest)"
echo "  $(date)"
echo "========================================================"

# Run pytest with uv
uv run python -m pytest tests/ $PYTEST_ARGS

echo ""
echo "========================================================"
echo "  Test suite completed!"
echo "  $(date)"
echo "========================================================"
