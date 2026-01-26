#!/bin/bash

# Default values
MODEL="Qwen/Qwen2.5-7B-Instruct" # Balanced model for L4 GPU (24GB)
PORT=8000
GPU_MEMORY_UTILIZATION=0.80
MAX_MODEL_LEN="" # Default to auto/empty

# Help function
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -m, --model <name>       Model name or path (default: $MODEL)"
    echo "  -p, --port <number>      Port to run the server on (default: $PORT)"
    echo "  --gpu-util <float>       GPU memory utilization (default: $GPU_MEMORY_UTILIZATION)"
    echo "  --max-len <int>          Max model length (default: Auto)"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --model Qwen/Qwen2.5-32B-Instruct --gpu-util 0.95"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        --gpu-util)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --max-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

echo "Starting vLLM server..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo "GPU Util: $GPU_MEMORY_UTILIZATION"
if [ -n "$MAX_MODEL_LEN" ]; then
    echo "Max Model Len: $MAX_MODEL_LEN"
else
    echo "Max Model Len: Auto"
fi

# Check if vllm is installed
if ! command -v vllm &> /dev/null; then
    echo "Error: vllm is not installed. Please run 'uv sync --extra vllm' or pip install vllm"
    exit 1
fi

# Run vLLM
CMD="vllm serve $MODEL --port $PORT --gpu-memory-utilization $GPU_MEMORY_UTILIZATION --trust-remote-code"

if [ -n "$MAX_MODEL_LEN" ]; then
    CMD="$CMD --max-model-len $MAX_MODEL_LEN"
fi

echo "Executing: $CMD"
eval $CMD
