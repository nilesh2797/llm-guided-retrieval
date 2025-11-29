#!/bin/bash

# Create log file with timestamp
mkdir -p ../logs
LOG_FILE="../logs/run_$(date '+%Y_%m_%d').log"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting run_all.sh script"

# Common params (key value pairs or flags). Run-specific params override these.
COMMON_PARAMS=(
    --suffix mse_calib_leaf_think-1
    --reasoning_in_traversal_prompt -1
    --load_existing
    --num_iters 20
    # --rerank
    # --llm_api_backend vllm
    # --llm Qwen/Qwen3-VL-8B-Instruct
    # --llm_api_staggering_delay 0.02
    # --llm_api_timeout 60
    # --llm_api_max_retries 3
)

# Define RUNS directly as strings (space-separated args)
RUNS=(
    # static corpus datasets
    "--subset biology --tree_version bottom-up"
    # "--subset economics --tree_version bottom-up"
    # "--subset earth_science --tree_version bottom-up"
    # "--subset psychology --tree_version bottom-up"
    # "--subset robotics --tree_version bottom-up"
    # "--subset stackoverflow --tree_version bottom-up"
    # "--subset sustainable_living --tree_version bottom-up"
    # "--subset theoremqa_theorems --tree_version top-down"
    # "--subset pony --tree_version bottom-up"
    # # dynamic corpus datasets
    # "--subset theoremqa_questions --tree_version top-down --num_iters 30 --relevance_chain_factor 0.75"
    # "--subset leetcode --tree_version top-down --num_iters 30 --relevance_chain_factor 0.75"
    # "--subset aops --tree_version top-down --num_iters 30 --relevance_chain_factor 0.75"
    )

# Collect option keys (those starting with --) from an array into an assoc set
collect_keys() {
    local -n _arr="$1"
    local -n _out="$2"
    local i=0
    while (( i < ${#_arr[@]} )); do
        local tok="${_arr[i]}"
        if [[ "$tok" == --* ]]; then
            _out["$tok"]=1
            if (( i+1 < ${#_arr[@]} )) && [[ "${_arr[i+1]}" != --* ]]; then
                ((i+=2))
                continue
            fi
        fi
        ((i++))
    done
}

# Iterate over iteration lists
for idx in "${!RUNS[@]}"; do
    iter_def="${RUNS[idx]}"

    # Parse the iteration string into an array
    read -r -a ITER_ARR <<< "$iter_def"

    # Build final args: first common params, then iteration-specific params
    final_args=()

    # Append COMMON_PARAMS first (preserve key-value groupings)
    i=0
    while (( i < ${#COMMON_PARAMS[@]} )); do
        key="${COMMON_PARAMS[i]}"
        final_args+=("$key")
        if (( i+1 < ${#COMMON_PARAMS[@]} )) && [[ "${COMMON_PARAMS[i+1]}" != --* ]]; then
            final_args+=("${COMMON_PARAMS[i+1]}")
            ((i+=2))
            continue
        fi
        ((i++))
    done

    # Append iteration-specific params (later args take precedence)
    final_args+=("${ITER_ARR[@]}")

    cmd=( python run.py "${final_args[@]}" )
    printf -v cmd_str '%q ' "${cmd[@]}"
    log "Executing: $cmd_str"

    "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log "Error in iteration: $((idx+1))"
        exit 1
    fi

    log "Completed iteration: $((idx+1))"
    log "---"
done

log "All RUNS completed successfully!"
