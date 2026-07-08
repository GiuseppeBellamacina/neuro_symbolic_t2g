#!/bin/bash
# ============================================================================
# Batch Test Runner — Neuro-Symbolic T2G
#
# Uso:
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

PASS_COUNT=0
FAIL_COUNT=0
FAILED_TESTS=""

run_test() {
    local script="$1"
    local label="$2"
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  Running: $label"
    echo "════════════════════════════════════════════════════════════════"
    if python "$script"; then
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_TESTS="$FAILED_TESTS  $label\n"
    fi
}

echo "========================================================"
echo "  Neuro-Symbolic T2G — Test Suite"
echo "  $(date)"
echo "========================================================"

# 1. Data ingestion (needs internet — can be skipped)
if [ "$SKIP_DATA" -eq 0 ]; then
    run_test "tests/test_data.py" "Data Ingestion & Transition Matrix"
else
    echo ""
    echo "  ⏭  Skipping data test (--skip-data)"
fi

# 2. Grammar & constrained decoding (tokenizer needed, no internet)
run_test "tests/test_grammar.py" "Grammar & Constrained Decoding"

# 3. Reward functions (needs initialized rewards)
run_test "tests/test_rewards.py" "Reward Functions"

# 4. Metrics & Utils (needs initialized rewards)
run_test "tests/test_metrics.py" "Metrics & Utils"

# 5. Integration (end-to-end coherence, needs internet)
if [ "$SKIP_DATA" -eq 0 ]; then
    run_test "tests/test_integration.py" "Integration & Coherence"
else
    echo ""
    echo "  ⏭  Skipping integration test (--skip-data)"
fi

# 6. Monitor parsing (pure logic, no deps)
run_test "tests/test_monitor.py" "Monitor Parsing Logic"

# 7. Config validation (YAML schema checks)
run_test "tests/validate_configs.py" "Config Validation"

echo ""
echo "========================================================"
echo "  TEST SUITE COMPLETE"
echo "  Passed: $PASS_COUNT | Failed: $FAIL_COUNT"
echo "========================================================"

if [ -n "$FAILED_TESTS" ]; then
    echo ""
    echo "FAILED:"
    echo -e "$FAILED_TESTS"
    exit 1
fi

echo "✅ All tests passed!"
exit 0
