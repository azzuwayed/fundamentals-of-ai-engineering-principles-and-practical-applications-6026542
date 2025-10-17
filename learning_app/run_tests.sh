#!/bin/bash

# ============================================================================
# AI Engineering Learning App - Interactive Test Runner
# ============================================================================
# This script provides an interactive menu to run test suites for the app.
#
# Usage: ./run_tests.sh
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Test files (located in tests/ directory)
TEST_FILES=(
    "tests/test_phase1_visualization_explainability.py"
    "tests/test_phase2_advanced_retrieval.py"
    "tests/test_phase3_rag_chat.py"
)

TEST_NAMES=(
    "Phase 1: Visualization & Explainability"
    "Phase 2: Advanced Retrieval"
    "Phase 3: RAG Chat"
)

# Helper functions
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${CYAN}→ $1${NC}"
}

print_test_header() {
    echo -e "\n${MAGENTA}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}║  $1${NC}"
    echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════════╝${NC}\n"
}

# ============================================================================
# Setup Environment
# ============================================================================
setup_environment() {
    cd "$PROJECT_ROOT"

    # Check if virtual environment exists
    if [ ! -d ".venv" ]; then
        print_error "Virtual environment not found"
        print_info "Please run ./learning_app/run.sh first to set up the environment"
        exit 1
    fi

    # Activate virtual environment
    source .venv/bin/activate

    # Set environment variables to suppress warnings
    export TOKENIZERS_PARALLELISM=false
    export OMP_MAX_ACTIVE_LEVELS=1  # Replaces deprecated OMP_NESTED
    export KMP_WARNINGS=0            # Suppress OpenMP runtime warnings

    cd "$SCRIPT_DIR"
}

# ============================================================================
# Run a single test file
# ============================================================================
run_test() {
    local test_file=$1
    local test_name=$2

    print_test_header "$test_name"

    if [ ! -f "$test_file" ]; then
        print_error "Test file not found: $test_file"
        return 1
    fi

    print_info "Running: python3 $test_file"
    echo ""

    # Run the test and capture exit code
    if python3 "$test_file"; then
        echo ""
        print_success "Test passed: $test_name"
        return 0
    else
        echo ""
        print_error "Test failed: $test_name"
        return 1
    fi
}

# ============================================================================
# Run all tests
# ============================================================================
run_all_tests() {
    print_header "Running All Tests"

    local total_tests=${#TEST_FILES[@]}
    local passed_tests=0
    local failed_tests=0
    local failed_names=()

    for i in "${!TEST_FILES[@]}"; do
        if run_test "${TEST_FILES[$i]}" "${TEST_NAMES[$i]}"; then
            ((passed_tests++))
        else
            ((failed_tests++))
            failed_names+=("${TEST_NAMES[$i]}")
        fi

        # Add separator between tests
        if [ $i -lt $((total_tests - 1)) ]; then
            echo ""
            echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"
        fi
    done

    # Summary
    echo ""
    print_header "Test Summary"
    echo -e "${BLUE}Total Tests:${NC}  $total_tests"
    echo -e "${GREEN}Passed:${NC}       $passed_tests"
    echo -e "${RED}Failed:${NC}       $failed_tests"

    if [ $failed_tests -gt 0 ]; then
        echo ""
        print_error "Failed Tests:"
        for name in "${failed_names[@]}"; do
            echo -e "  ${RED}✗${NC} $name"
        done
        return 1
    else
        echo ""
        print_success "All tests passed!"
        return 0
    fi
}

# ============================================================================
# Interactive Menu
# ============================================================================
show_menu() {
    clear
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  AI Engineering Learning App - Test Runner                    ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${CYAN}Available Tests:${NC}"
    echo ""

    for i in "${!TEST_FILES[@]}"; do
        echo -e "  ${GREEN}$((i+1)).${NC} ${TEST_NAMES[$i]}"
        echo -e "     ${CYAN}→${NC} ${TEST_FILES[$i]}"
        echo ""
    done

    echo -e "${CYAN}Options:${NC}"
    echo ""
    echo -e "  ${GREEN}a${NC}. Run all tests"
    echo -e "  ${GREEN}1-${#TEST_FILES[@]}${NC}. Run specific test"
    echo -e "  ${GREEN}q${NC}. Quit"
    echo ""
    echo -n -e "${YELLOW}Enter your choice:${NC} "
}

# ============================================================================
# Main
# ============================================================================
main() {
    # Setup
    setup_environment

    # Interactive loop
    while true; do
        show_menu
        read -r choice

        case $choice in
            a|A)
                if run_all_tests; then
                    exit_code=0
                else
                    exit_code=1
                fi
                ;;
            [1-9])
                if [ "$choice" -le "${#TEST_FILES[@]}" ]; then
                    index=$((choice - 1))
                    if run_test "${TEST_FILES[$index]}" "${TEST_NAMES[$index]}"; then
                        exit_code=0
                    else
                        exit_code=1
                    fi
                else
                    print_error "Invalid test number"
                    exit_code=1
                fi
                ;;
            q|Q)
                print_info "Exiting test runner"
                exit 0
                ;;
            *)
                print_error "Invalid choice. Please try again."
                exit_code=1
                ;;
        esac

        # Pause before showing menu again
        echo ""
        echo -n -e "${YELLOW}Press Enter to continue...${NC}"
        read -r
    done
}

# Run main function
main
