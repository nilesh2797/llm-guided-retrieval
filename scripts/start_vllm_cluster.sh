#!/bin/bash
# Script to start vLLM servers in data parallel or tensor parallel mode
#
# Usage:
#   ./start_vllm_cluster.sh [MODEL] [MODE]
#
# Examples:
#   ./start_vllm_cluster.sh                                    # Data parallel with default model
#   ./start_vllm_cluster.sh "Qwen/Qwen3-VL-8B-Instruct" data   # Data parallel
#   ./start_vllm_cluster.sh "meta-llama/Llama-2-70b-hf" tensor # Tensor parallel
#
# Modes:
#   data   - Run multiple servers (one per GPU) for maximum throughput
#   tensor - Run single server with model split across GPUs for large models

set -e

# Configuration
MODEL="${1:-Qwen/Qwen3-VL-8B-Instruct}"
MODE="${2:-data}"  # "data" or "tensor"
BASE_PORT=8000
NUM_GPUS=4
GPU_MEM_UTIL=0.95
# Specify GPU IDs to use
GPU_IDS=(0 1 3 4)
MAX_MODEL_LEN=128000

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Validate mode
if [[ "$MODE" != "data" && "$MODE" != "tensor" ]]; then
    echo -e "${RED}Error: MODE must be 'data' or 'tensor', got: $MODE${NC}"
    echo "Usage: $0 [MODEL] [MODE]"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Save mode to file for stop script
echo "$MODE" > logs/cluster_mode.txt

if [[ "$MODE" == "tensor" ]]; then
    # ============================================
    # TENSOR PARALLEL MODE
    # ============================================
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}Starting vLLM Tensor Parallel Server${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "Model: ${MODEL}"
    echo -e "GPUs: ${GPU_IDS[@]}"
    echo -e "Tensor Parallel Size: ${NUM_GPUS}"
    echo -e "Port: ${BASE_PORT}"
    echo -e "${BLUE}================================================${NC}\n"

    # Build GPU list as comma-separated string
    GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
    LOG_FILE="logs/vllm_tensor_port${BASE_PORT}.log"

    echo -e "${GREEN}Starting tensor parallel server on GPUs: $GPU_LIST${NC}"

    CUDA_VISIBLE_DEVICES=$GPU_LIST nohup python -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --port $BASE_PORT \
        --host 0.0.0.0 \
        --tensor-parallel-size $NUM_GPUS \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --disable-log-requests \
        --max-model-len $MAX_MODEL_LEN \
        > $LOG_FILE 2>&1 &

    PID=$!
    echo $PID > "logs/vllm_tensor.pid"
    echo -e "${GREEN}Server started with PID: ${PID}${NC}"
    echo -e "Log file: ${LOG_FILE}\n"

    echo -e "\nWaiting for server to initialize..."
    sleep 15

    # Check server health
    echo -e "\n${BLUE}Checking server health...${NC}"
    if curl -s http://localhost:${BASE_PORT}/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Tensor parallel server (port ${BASE_PORT}): HEALTHY"
    else
        echo -e "${YELLOW}⚠${NC} Tensor parallel server (port ${BASE_PORT}): Still initializing..."
    fi

    echo -e "\n${BLUE}================================================${NC}"
    echo -e "${GREEN}Tensor Parallel Server Ready!${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "\nUsage in Python:"
    echo -e '  client = VllmAPI('
    echo -e '      model_name="'$MODEL'",'
    echo -e '      base_url="http://localhost:'$BASE_PORT'/v1"'
    echo -e '  )'
    echo -e "\nNote: Tensor parallelism provides lower throughput but enables larger models."
    echo -e "For maximum throughput with smaller models, use data parallel mode.\n"

else
    # ============================================
    # DATA PARALLEL MODE
    # ============================================
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}Starting vLLM Data Parallel Cluster${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "Model: ${MODEL}"
    echo -e "GPUs: ${GPU_IDS[@]}"
    echo -e "Ports: ${BASE_PORT}-$((BASE_PORT + NUM_GPUS - 1))"
    echo -e "${BLUE}================================================${NC}\n"

    # Start servers
    for i in $(seq 0 $((NUM_GPUS-1))); do
        GPU_ID=${GPU_IDS[$i]}
        PORT=$((BASE_PORT + i))
        LOG_FILE="logs/vllm_gpu${i}_port${PORT}.log"

        echo -e "${GREEN}[GPU $GPU_ID]${NC} Starting vLLM server on port ${PORT}..."

        CUDA_VISIBLE_DEVICES=$GPU_ID nohup python -m vllm.entrypoints.openai.api_server \
            --model $MODEL \
            --port $PORT \
            --host 0.0.0.0 \
            --gpu-memory-utilization $GPU_MEM_UTIL \
            --disable-log-requests \
            --max-model-len $MAX_MODEL_LEN \
            > $LOG_FILE 2>&1 &

        PID=$!
        echo $PID > "logs/vllm_gpu${i}.pid"
        echo -e "${GREEN}[GPU $GPU_ID]${NC} Server started with PID: ${PID}"
        echo -e "        Log file: ${LOG_FILE}\n"

        # Give each server time to initialize
        sleep 3
    done

    echo -e "\n${BLUE}================================================${NC}"
    echo -e "${GREEN}All vLLM servers started successfully!${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "URLs:"
    for i in $(seq 0 $((NUM_GPUS-1))); do
        GPU_ID=${GPU_IDS[$i]}
        PORT=$((BASE_PORT + i))
        echo -e "  GPU $GPU_ID: http://localhost:${PORT}/v1"
    done

    echo -e "\nWaiting for servers to fully initialize..."
    sleep 10

    # Check server health
    echo -e "\n${BLUE}Checking server health...${NC}"
    for i in $(seq 0 $((NUM_GPUS-1))); do
        GPU_ID=${GPU_IDS[$i]}
        PORT=$((BASE_PORT + i))
        if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} GPU $GPU_ID (port ${PORT}): HEALTHY"
        else
            echo -e "${YELLOW}⚠${NC} GPU $GPU_ID (port ${PORT}): Still initializing..."
        fi
    done

    echo -e "\n${BLUE}================================================${NC}"
    echo -e "${GREEN}Data Parallel Cluster Ready!${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "\nUsage in Python:"
    echo -e '  client = VllmAPI('
    echo -e '      model_name="'$MODEL'",'
    echo -e '      base_url="http://localhost:8000/v1,http://localhost:8001/v1,http://localhost:8002/v1,http://localhost:8003/v1"'
    echo -e '  )'
    echo -e "\nTo test: python test_vllm_cluster.py"
fi

echo -e "\nTo stop: ./stop_vllm_cluster.sh"
echo -e "View logs: tail -f logs/vllm*.log\n"
