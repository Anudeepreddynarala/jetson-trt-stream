#!/bin/bash
#
# YOLO-Rust-TRT Launch Script
# Starts Rust networking layer and Python inference pipeline
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
RUST_BIN="$PROJECT_ROOT/rust_networking/target/release/yolo-rust-networking"
PYTHON_MAIN="$PROJECT_ROOT/inference/main.py"
CONFIG_FILE="$PROJECT_ROOT/config/config.yaml"
LOG_DIR="$PROJECT_ROOT/logs"

# PIDs for cleanup
RUST_PID=""
PYTHON_PID=""

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"

    if [ -n "$PYTHON_PID" ] && kill -0 $PYTHON_PID 2>/dev/null; then
        echo "Stopping Python pipeline (PID: $PYTHON_PID)"
        kill -TERM $PYTHON_PID 2>/dev/null || true
        wait $PYTHON_PID 2>/dev/null || true
    fi

    if [ -n "$RUST_PID" ] && kill -0 $RUST_PID 2>/dev/null; then
        echo "Stopping Rust server (PID: $RUST_PID)"
        kill -TERM $RUST_PID 2>/dev/null || true
        wait $RUST_PID 2>/dev/null || true
    fi

    # Cleanup Unix socket
    if [ -S "/tmp/yolo_rust.sock" ]; then
        rm -f /tmp/yolo_rust.sock
    fi

    echo -e "${GREEN}✓ Shutdown complete${NC}"
    exit 0
}

# Register cleanup on exit
trap cleanup SIGINT SIGTERM EXIT

# Print banner
echo "============================================================"
echo "  YOLO-Rust-TRT Pipeline Launcher"
echo "============================================================"
echo ""

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

# Check Rust binary
if [ ! -f "$RUST_BIN" ]; then
    echo -e "${YELLOW}⚠ Rust binary not found, building...${NC}"
    cd "$PROJECT_ROOT/rust_networking"
    cargo build --release
    cd "$PROJECT_ROOT"
fi

if [ ! -f "$RUST_BIN" ]; then
    echo -e "${RED}✗ Failed to build Rust binary${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Rust binary ready"

# Check Python script
if [ ! -f "$PYTHON_MAIN" ]; then
    echo -e "${RED}✗ Python main script not found: $PYTHON_MAIN${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Python pipeline ready"

# Check config
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}✗ Configuration file not found: $CONFIG_FILE${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Configuration file found"

# Check models
MODEL_PATH=$(grep "model_path:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')
FULL_MODEL_PATH="$PROJECT_ROOT/$MODEL_PATH"

if [ ! -f "$FULL_MODEL_PATH" ]; then
    echo -e "${RED}✗ Model not found: $FULL_MODEL_PATH${NC}"
    echo -e "${YELLOW}Run: python3 scripts/export_model.py && python3 scripts/convert_to_trt.py${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Model found: $MODEL_PATH"

# Create log directory
mkdir -p "$LOG_DIR"
echo -e "${GREEN}✓${NC} Log directory ready"

echo ""

# Parse command line arguments
START_MODE="both"  # Default: start both Rust and Python

case "${1:-}" in
    --rust-only)
        START_MODE="rust"
        ;;
    --python-only)
        START_MODE="python"
        ;;
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --rust-only     Start only Rust networking layer"
        echo "  --python-only   Start only Python inference pipeline"
        echo "  --help, -h      Show this help message"
        echo ""
        echo "Default: Start both Rust and Python"
        exit 0
        ;;
esac

# Start Rust networking layer
if [ "$START_MODE" = "both" ] || [ "$START_MODE" = "rust" ]; then
    echo -e "${BLUE}Starting Rust networking layer...${NC}"
    cd "$PROJECT_ROOT/rust_networking"
    "$RUST_BIN" > "$LOG_DIR/rust_networking.log" 2>&1 &
    RUST_PID=$!
    cd "$PROJECT_ROOT"

    # Wait a bit for Rust to initialize
    sleep 1

    # Check if Rust started successfully
    if ! kill -0 $RUST_PID 2>/dev/null; then
        echo -e "${RED}✗ Rust server failed to start${NC}"
        cat "$LOG_DIR/rust_networking.log"
        exit 1
    fi

    echo -e "${GREEN}✓${NC} Rust server started (PID: $RUST_PID)"
    echo -e "  Unix socket: /tmp/yolo_rust.sock"
    echo -e "  TCP port: 8888"
    echo -e "  UDP target: 127.0.0.1:9999"
fi

# Start Python pipeline
if [ "$START_MODE" = "both" ] || [ "$START_MODE" = "python" ]; then
    echo -e "${BLUE}Starting Python inference pipeline...${NC}"

    cd "$PROJECT_ROOT"
    python3 "$PYTHON_MAIN" > "$LOG_DIR/python_pipeline.log" 2>&1 &
    PYTHON_PID=$!

    # Wait a bit for Python to initialize
    sleep 2

    # Check if Python started successfully
    if ! kill -0 $PYTHON_PID 2>/dev/null; then
        echo -e "${RED}✗ Python pipeline failed to start${NC}"
        cat "$LOG_DIR/python_pipeline.log"
        exit 1
    fi

    echo -e "${GREEN}✓${NC} Python pipeline started (PID: $PYTHON_PID)"
fi

echo ""
echo "============================================================"
echo -e "${GREEN}Pipeline is running!${NC}"
echo "============================================================"
echo ""
echo "Logs:"
if [ -n "$RUST_PID" ]; then
    echo "  Rust:   tail -f $LOG_DIR/rust_networking.log"
fi
if [ -n "$PYTHON_PID" ]; then
    echo "  Python: tail -f $LOG_DIR/python_pipeline.log"
fi
echo ""
echo "TCP Commands (port 8888):"
echo "  Stats:  echo '{\"command\":\"STATS\"}' | nc 127.0.0.1 8888"
echo "  Config: echo '{\"command\":\"GET_CONFIG\"}' | nc 127.0.0.1 8888"
echo ""
echo "UDP Test (port 9999):"
echo "  python3 scripts/udp_test_client.py"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

# Wait for processes
if [ -n "$PYTHON_PID" ]; then
    wait $PYTHON_PID
elif [ -n "$RUST_PID" ]; then
    wait $RUST_PID
fi
