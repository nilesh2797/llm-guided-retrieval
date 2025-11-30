#!/bin/bash
# Script to stop vLLM servers (data parallel or tensor parallel)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Determine mode from saved file
MODE="data"  # default
if [ -f "logs/cluster_mode.txt" ]; then
    MODE=$(cat logs/cluster_mode.txt)
fi

echo -e "${BLUE}================================================${NC}"
if [[ "$MODE" == "tensor" ]]; then
    echo -e "${BLUE}Stopping vLLM Tensor Parallel Server${NC}"
else
    echo -e "${BLUE}Stopping vLLM Data Parallel Cluster${NC}"
fi
echo -e "${BLUE}================================================${NC}\n"

NUM_GPUS=4
STOPPED=0

if [[ "$MODE" == "tensor" ]]; then
    # ============================================
    # STOP TENSOR PARALLEL SERVER
    # ============================================
    PID_FILE="logs/vllm_tensor.pid"

    if [ -f "$PID_FILE" ]; then
        PID=$(cat $PID_FILE)
        if kill -0 $PID 2>/dev/null; then
            echo -e "${GREEN}Stopping tensor parallel server (PID: ${PID})...${NC}"
            kill $PID
            STOPPED=1
            rm $PID_FILE
            echo -e "${GREEN}✓ Server stopped${NC}"
        else
            echo -e "${RED}Server not running (stale PID file)${NC}"
            rm $PID_FILE
        fi
    else
        echo -e "${YELLOW}No PID file found for tensor parallel server${NC}"
    fi

else
    # ============================================
    # STOP DATA PARALLEL SERVERS
    # ============================================
    # Stop servers using PID files
    for i in $(seq 0 $((NUM_GPUS-1))); do
        PID_FILE="logs/vllm_gpu${i}.pid"

        if [ -f "$PID_FILE" ]; then
            PID=$(cat $PID_FILE)
            if kill -0 $PID 2>/dev/null; then
                echo -e "${GREEN}[GPU $i]${NC} Stopping server (PID: ${PID})..."
                kill $PID
                STOPPED=$((STOPPED + 1))
                rm $PID_FILE
            else
                echo -e "${RED}[GPU $i]${NC} Server not running (stale PID file)"
                rm $PID_FILE
            fi
        else
            echo -e "${RED}[GPU $i]${NC} No PID file found"
        fi
    done
fi

# Also try to kill any remaining vllm processes
echo -e "\n${BLUE}Checking for remaining vLLM processes...${NC}"
REMAINING=$(pgrep -f "vllm.entrypoints.openai.api_server" | wc -l)

if [ $REMAINING -gt 0 ]; then
    echo -e "${GREEN}Found ${REMAINING} remaining process(es)${NC}"
    pkill -f "vllm.entrypoints.openai.api_server"
    sleep 2
    echo -e "${GREEN}Cleaned up remaining processes${NC}"
fi

# Clean up mode file
if [ -f "logs/cluster_mode.txt" ]; then
    rm logs/cluster_mode.txt
fi

echo -e "\n${BLUE}================================================${NC}"
if [ $STOPPED -gt 0 ]; then
    echo -e "${GREEN}Successfully stopped ${STOPPED} server(s)${NC}"
else
    echo -e "${RED}No servers were running${NC}"
fi
echo -e "${BLUE}================================================${NC}\n"
