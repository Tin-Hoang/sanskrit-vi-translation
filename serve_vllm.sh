#!/bin/bash

# Default values (run on NVIDIA L4 - 24GB)
MODEL="Qwen/Qwen2.5-32B-Instruct-AWQ"
PORT=8000
GPU_MEMORY_UTILIZATION=0.92
MAX_MODEL_LEN=2048
DTYPE="auto"
QUANTIZATION="awq"
KV_CACHE_DTYPE="auto"

# Help function
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -m, --model <name>       Model name or path (default: $MODEL)"
    echo "  -p, --port <number>      Port to run the server on (default: $PORT)"
    echo "  --gpu-util <float>       GPU memory utilization (default: $GPU_MEMORY_UTILIZATION)"
    echo "  --max-len <int>          Max model length (default: Auto)"
    echo "  --dtype <type>           Data type (auto, float16, bfloat16) (default: $DTYPE)"
    echo "  -q, --quantization <alg> Quantization (awq, gptq, squeezellm) (default: None)"
    echo "  --kv-cache-dtype <type>  KV Cache data type (auto, fp8) (default: $KV_CACHE_DTYPE)"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --model Qwen/Qwen2.5-32B-Instruct-AWQ --quantization awq"
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
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        -q|--quantization)
            QUANTIZATION="$2"
            shift 2
            ;;
        --kv-cache-dtype)
            KV_CACHE_DTYPE="$2"
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
echo "Dtype: $DTYPE"

if [ -n "$MAX_MODEL_LEN" ]; then
    echo "Max Model Len: $MAX_MODEL_LEN"
else
    echo "Max Model Len: Auto"
fi

if [ -n "$QUANTIZATION" ]; then
    echo "Quantization: $QUANTIZATION"
fi

# Check if vllm is installed
if ! command -v vllm &> /dev/null; then
    echo "Error: vllm is not installed. Please run 'uv sync --extra vllm' or pip install vllm"
    exit 1
fi

# Export PyTorch Allocator Config to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run vLLM
# - enable-prefix-caching: Improves performance for system prompts/multi-turn
# - trust-remote-code: Required for some models
CMD="vllm serve $MODEL \
    --port $PORT \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --dtype $DTYPE \
    --kv-cache-dtype $KV_CACHE_DTYPE \
    --max-num-seqs 32 \
    --trust-remote-code"

if [ -n "$MAX_MODEL_LEN" ]; then
    CMD="$CMD --max-model-len $MAX_MODEL_LEN"
fi

if [ -n "$QUANTIZATION" ]; then
    CMD="$CMD --quantization $QUANTIZATION"
fi

echo "Executing: $CMD"
eval $CMD
