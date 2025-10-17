#!/bin/bash

# ============================================================================
# AI Engineering Learning App - Launch Script
# ============================================================================
# This script automatically sets up and launches the Gradio learning app
# with comprehensive environment verification and dependency management.
#
# Usage: ./run.sh [--force-reinstall] [--port PORT]
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
DEFAULT_PORT=7860
REQUIRED_PYTHON_MAJOR=3
REQUIRED_PYTHON_MINOR=12
FORCE_REINSTALL=false
PORT=$DEFAULT_PORT

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force-reinstall)
            FORCE_REINSTALL=true
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --force-reinstall    Force reinstall all dependencies"
            echo "  --port PORT          Run on custom port (default: 7860)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}✗ Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

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
    echo -e "${BLUE}→ $1${NC}"
}

# ============================================================================
# STEP 1: Check Python Version
# ============================================================================
print_header "Step 1: Checking Python Version"

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    print_info "Please install Python 3.12 or higher"
    print_info "Visit: https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

print_info "Found Python $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt "$REQUIRED_PYTHON_MAJOR" ] || \
   ([ "$PYTHON_MAJOR" -eq "$REQUIRED_PYTHON_MAJOR" ] && [ "$PYTHON_MINOR" -lt "$REQUIRED_PYTHON_MINOR" ]); then
    print_error "Python $REQUIRED_PYTHON_MAJOR.$REQUIRED_PYTHON_MINOR or higher is required"
    print_info "Current version: $PYTHON_VERSION"
    exit 1
fi

print_success "Python version is compatible"

# ============================================================================
# STEP 2: Check for uv Package Manager
# ============================================================================
print_header "Step 2: Checking Package Manager"

if ! command -v uv &> /dev/null; then
    print_error "uv package manager is not installed"
    print_info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Reload PATH
    export PATH="$HOME/.local/bin:$PATH"

    if ! command -v uv &> /dev/null; then
        print_error "Failed to install uv"
        print_info "Please install uv manually: https://github.com/astral-sh/uv"
        exit 1
    fi
    print_success "uv installed successfully"
else
    print_success "uv package manager found"
fi

# ============================================================================
# STEP 3: Navigate to Project Directory
# ============================================================================
print_header "Step 3: Setting Up Environment"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

print_info "Script location: $SCRIPT_DIR"
print_info "Project root: $PROJECT_ROOT"

cd "$PROJECT_ROOT"

# ============================================================================
# STEP 4: Virtual Environment Setup
# ============================================================================
print_header "Step 4: Virtual Environment Setup"

if [ ! -d ".venv" ]; then
    print_info "Creating virtual environment..."
    uv venv .venv --python 3.12
    print_success "Virtual environment created"
else
    print_success "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source .venv/bin/activate
print_success "Virtual environment activated"

# ============================================================================
# STEP 5: Install/Update Dependencies
# ============================================================================
print_header "Step 5: Installing Dependencies"

if [ "$FORCE_REINSTALL" = true ]; then
    print_warning "Force reinstall requested - removing existing packages..."
    uv pip uninstall -y gradio 2>/dev/null || true
fi

print_info "Installing requirements..."
UV_LINK_MODE=copy uv pip install -r requirements.txt

print_success "Dependencies installed"

# ============================================================================
# STEP 6: Verify Gradio Installation
# ============================================================================
print_header "Step 6: Verifying Gradio Installation"

GRADIO_VERSION=$(python3 -c "import gradio; print(gradio.__version__)" 2>/dev/null || echo "not_found")

if [ "$GRADIO_VERSION" = "not_found" ]; then
    print_error "Gradio is not installed properly"
    exit 1
fi

print_success "Gradio version: $GRADIO_VERSION"

EXPECTED_VERSION="5.49.1"
if [ "$GRADIO_VERSION" != "$EXPECTED_VERSION" ]; then
    print_warning "Expected Gradio version $EXPECTED_VERSION, found $GRADIO_VERSION"
fi

# ============================================================================
# STEP 7: Verify App Dependencies
# ============================================================================
print_header "Step 7: Verifying App Dependencies"

print_info "Testing imports..."

cd "$SCRIPT_DIR"

python3 << 'EOF'
import sys
import os

sys.path.insert(0, '.')

try:
    print("  → Testing Gradio import...", end=" ")
    import gradio as gr
    print("✓")

    print("  → Testing modules import...", end=" ")
    from modules import (
        DocumentProcessor,
        EmbeddingsEngine,
        VectorStore,
        RetrievalPipeline
    )
    print("✓")

    print("  → Testing Phase 1 modules...", end=" ")
    from modules.visualization_engine import VisualizationEngine
    from modules.explainability_engine import ExplainabilityEngine
    print("✓")

    print("  → Testing utils import...", end=" ")
    from utils import (
        format_results_table,
        format_metrics,
        validate_text_input
    )
    print("✓")

    print("  → Testing Phase 1 utils...", end=" ")
    from utils.plot_helpers import create_plotly_config
    print("✓")

    print("  → Testing Phase 2 modules...", end=" ")
    from modules.query_intelligence import QueryIntelligence
    from modules.multi_query_engine import MultiQueryEngine
    from modules.advanced_filtering import AdvancedFilter
    print("✓")

    print("  → Testing Phase 3 modules...", end=" ")
    from modules.llm_manager import LLMManager, LLMConfig
    from modules.context_manager import ContextManager
    from modules.rag_pipeline import RAGPipeline
    from modules.conversation_engine import ConversationEngine
    print("✓")

    print("\n✓ All imports successful (including Phase 1, Phase 2 & Phase 3 enhancements)")
    sys.exit(0)

except Exception as e:
    print(f"\n✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    print_error "Dependency verification failed"
    exit 1
fi

print_success "All dependencies verified"

# ============================================================================
# STEP 8: Check Port Availability
# ============================================================================
print_header "Step 8: Checking Port Availability"

if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    print_warning "Port $PORT is already in use"
    print_info "Attempting to find available port..."

    # Try alternative ports
    for alt_port in {7861..7870}; do
        if ! lsof -Pi :$alt_port -sTCP:LISTEN -t >/dev/null 2>&1; then
            PORT=$alt_port
            print_success "Using alternative port: $PORT"
            break
        fi
    done

    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_error "No available ports found in range 7860-7870"
        print_info "Please stop other services or specify a custom port with --port"
        exit 1
    fi
else
    print_success "Port $PORT is available"
fi

# ============================================================================
# STEP 9: Launch Application
# ============================================================================
print_header "Step 9: Launching Application"

print_success "All checks passed!"
echo ""
print_info "Starting AI Engineering Learning App..."
print_info "Server will be available at: http://localhost:$PORT"
print_info "Press Ctrl+C to stop the server"
echo ""

# Export port for the app if needed
export GRADIO_SERVER_PORT=$PORT

# Launch the app
cd "$SCRIPT_DIR"
python3 app.py

# This will only execute if the app exits
print_info "Application stopped"
