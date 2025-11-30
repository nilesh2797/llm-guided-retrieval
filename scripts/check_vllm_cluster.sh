#!/bin/bash
# Script to check the status of vLLM servers (data parallel or tensor parallel)

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

BASE_PORT=8000
NUM_GPUS=4

# Determine mode from saved file
MODE="data"  # default
if [ -f "logs/cluster_mode.txt" ]; then
    MODE=$(cat logs/cluster_mode.txt)
fi

echo -e "${BLUE}================================================${NC}"
if [[ "$MODE" == "tensor" ]]; then
    echo -e "${BLUE}vLLM Tensor Parallel Server Status${NC}"
else
    echo -e "${BLUE}vLLM Data Parallel Cluster Status${NC}"
fi
echo -e "${BLUE}================================================${NC}\n"

HEALTHY=0
UNHEALTHY=0

if [[ "$MODE" == "tensor" ]]; then
    # ============================================
    # CHECK TENSOR PARALLEL SERVER
    # ============================================
    PORT=$BASE_PORT
    PID_FILE="logs/vllm_tensor.pid"

    echo -e "${BLUE}[Tensor Parallel Server - Port ${PORT}]${NC}"

    # Check PID
    if [ -f "$PID_FILE" ]; then
        PID=$(cat $PID_FILE)
        if kill -0 $PID 2>/dev/null; then
            echo -e "  Process: ${GREEN}Running${NC} (PID: ${PID})"
        else
            echo -e "  Process: ${RED}Not running${NC} (stale PID)"
        fi
    else
        echo -e "  Process: ${RED}No PID file${NC}"
    fi

    # Check health endpoint
    HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${PORT}/health 2>/dev/null)

    if [ "$HEALTH_RESPONSE" = "200" ]; then
        echo -e "  Health:  ${GREEN}HEALTHY${NC}"
        HEALTHY=1

        # Get model info
        MODEL_INFO=$(curl -s http://localhost:${PORT}/v1/models 2>/dev/null | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
        if [ ! -z "$MODEL_INFO" ]; then
            echo -e "  Model:   ${MODEL_INFO}"
        fi
    else
        echo -e "  Health:  ${RED}UNAVAILABLE${NC} (HTTP ${HEALTH_RESPONSE:-timeout})"
        UNHEALTHY=1
    fi

    # Check log for errors
    LOG_FILE="logs/vllm_tensor_port${PORT}.log"
    if [ -f "$LOG_FILE" ]; then
        ERROR_COUNT=$(grep -i "error\|exception" $LOG_FILE 2>/dev/null | wc -l)
        if [ $ERROR_COUNT -gt 0 ]; then
            echo -e "  Logs:    ${YELLOW}${ERROR_COUNT} errors/exceptions found${NC}"
            echo -e "           tail -n 50 ${LOG_FILE}"
        else
            echo -e "  Logs:    ${GREEN}No errors${NC}"
        fi
    fi

    echo ""

else
    # ============================================
    # CHECK DATA PARALLEL SERVERS
    # ============================================
    for i in $(seq 0 $((NUM_GPUS-1))); do
        PORT=$((BASE_PORT + i))
        PID_FILE="logs/vllm_gpu${i}.pid"

        echo -e "${BLUE}[GPU $i - Port ${PORT}]${NC}"

        # Check PID
        if [ -f "$PID_FILE" ]; then
            PID=$(cat $PID_FILE)
            if kill -0 $PID 2>/dev/null; then
                echo -e "  Process: ${GREEN}Running${NC} (PID: ${PID})"
            else
                echo -e "  Process: ${RED}Not running${NC} (stale PID)"
            fi
        else
            echo -e "  Process: ${RED}No PID file${NC}"
        fi

        # Check health endpoint
        HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${PORT}/health 2>/dev/null)

        if [ "$HEALTH_RESPONSE" = "200" ]; then
            echo -e "  Health:  ${GREEN}HEALTHY${NC}"
            HEALTHY=$((HEALTHY + 1))

            # Get model info
            MODEL_INFO=$(curl -s http://localhost:${PORT}/v1/models 2>/dev/null | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
            if [ ! -z "$MODEL_INFO" ]; then
                echo -e "  Model:   ${MODEL_INFO}"
            fi
        else
            echo -e "  Health:  ${RED}UNAVAILABLE${NC} (HTTP ${HEALTH_RESPONSE:-timeout})"
            UNHEALTHY=$((UNHEALTHY + 1))
        fi

        # Check log for errors
        LOG_FILE="logs/vllm_gpu${i}_port${PORT}.log"
        if [ -f "$LOG_FILE" ]; then
            ERROR_COUNT=$(grep -i "error\|exception" $LOG_FILE 2>/dev/null | wc -l)
            if [ $ERROR_COUNT -gt 0 ]; then
                echo -e "  Logs:    ${YELLOW}${ERROR_COUNT} errors/exceptions found${NC}"
                echo -e "           tail -n 50 ${LOG_FILE}"
            else
                echo -e "  Logs:    ${GREEN}No errors${NC}"
            fi
        fi

        echo ""
    done
fi

echo -e "${BLUE}================================================${NC}"
echo -e "Summary: ${GREEN}${HEALTHY} Healthy${NC} | ${RED}${UNHEALTHY} Unhealthy${NC}"
echo -e "${BLUE}================================================${NC}\n"

# GPU status
if command -v nvidia-smi &> /dev/null; then
    echo -e "${BLUE}GPU Utilization:${NC}"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    while IFS=',' read -r idx name util mem_used mem_total; do
        echo -e "  GPU ${idx}: ${name} - ${util}% util, ${mem_used}/${mem_total} MB"
    done
    echo ""
fi

# Quick performance test
if [ $HEALTHY -gt 0 ]; then
    echo -e "${BLUE}Quick Response Test:${NC}"
    PORT=$((BASE_PORT))
    START=$(date +%s%N)
    RESPONSE=$(curl -s -X POST http://localhost:${PORT}/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "dummy", "prompt": "test", "max_tokens": 1}' 2>/dev/null)
    END=$(date +%s%N)
    DURATION=$(( (END - START) / 1000000 ))

    if [ $? -eq 0 ]; then
        echo -e "  Server response time: ${DURATION}ms"
    else
        echo -e "  ${RED}Failed to get response${NC}"
    fi
fi

echo ""
