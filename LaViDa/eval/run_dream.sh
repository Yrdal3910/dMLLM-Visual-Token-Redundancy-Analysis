#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# The selected compression implementation must be named "llava" under eval/
# before this script starts.
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

MODEL_PATHS=(
    "jacklishufan/lavida-dream-v1.0-instruct"
)

OUTPUT_PATH="${OUTPUT_PATH:-exp/lavida_eval}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-llava}"
TASK_NAMES="${TASK_NAMES:-mme}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES

IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_IDS[@]}}"
BASE_PORT="${BASE_PORT:-29500}"

if (( NUM_PROCESSES < 1 || NUM_PROCESSES > ${#GPU_IDS[@]} )); then
    echo "NUM_PROCESSES=$NUM_PROCESSES is incompatible with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
    exit 2
fi

IFS=',' read -ra TASKS <<< "$TASK_NAMES"

declare -a TASK_QUEUE=()
for model_path in "${MODEL_PATHS[@]}"; do
    for task in "${TASKS[@]}"; do
        case "$task" in
            mmmu_val|mmmu_pro_standard|mmstar|ai2d|seedbench|mmbench_en_dev|mmmu_pro_vision|muirbench|videomme|mlvu_dev|mme|realworldqa)
                gen_kwargs='{"temperature":0,"cfg":0,"remasking":"low_confidence","max_new_tokens":4,"block_length":4,"step_per_block":2,"think_mode":"no_think","alg":"topk_margin","prefix_lm":"True"}'
                ;;
            chartqa)
                gen_kwargs='{"temperature":0,"cfg":0,"remasking":"low_confidence","max_new_tokens":16,"block_length":16,"step_per_block":8,"stopping_criteria":["\n"],"think_mode":"no_think","alg":"topk_margin","prefix_lm":"True"}'
                ;;
            docvqa_val|infovqa_val)
                gen_kwargs='{"temperature":0,"cfg":0,"remasking":"low_confidence","max_new_tokens":32,"block_length":32,"step_per_block":16,"think_mode":"no_think","alg":"topk_margin","prefix_lm":"True"}'
                ;;
            mathvista_testmini)
                gen_kwargs='{"temperature":0,"cfg":0,"remasking":"low_confidence","max_new_tokens":96,"block_length":96,"step_per_block":48,"think_mode":"think","alg":"topk_margin","prefix_lm":"True"}'
                ;;
            mathverse_testmini_vision)
                gen_kwargs='{"temperature":0,"cfg":0,"remasking":"low_confidence","max_new_tokens":64,"block_length":64,"step_per_block":32,"think_mode":"think","alg":"topk_margin","prefix_lm":"True"}'
                ;;
            detailcaps|video_dc499|short_test)
                gen_kwargs='{"temperature":0,"cfg":0,"remasking":"low_confidence","max_new_tokens":128,"block_length":128,"step_per_block":64,"think_mode":"think","alg":"topk_margin","prefix_lm":"True"}'
                ;;
            *)
                gen_kwargs='{"temperature":0,"cfg":0,"remasking":"low_confidence","max_new_tokens":4,"block_length":4,"step_per_block":2,"think_mode":"no_think","alg":"topk_margin","prefix_lm":"True"}'
                ;;
        esac

        TASK_QUEUE+=("$model_path"$'\t'"$task"$'\t'"$gen_kwargs")
    done
done

TOTAL_TASKS=${#TASK_QUEUE[@]}
FAILED_TASKS=0

echo "Running $TOTAL_TASKS evaluation task(s) with $NUM_PROCESSES process(es) on GPU(s): $CUDA_VISIBLE_DEVICES"

for ((i=0; i<TOTAL_TASKS; i++)); do
    IFS=$'\t' read -r MODEL_PATH TASK_NAME GEN_KWARGS <<< "${TASK_QUEUE[$i]}"

    MODEL_PATH_LAST="$(basename "$MODEL_PATH")"
    CURRENT_OUTPUT_PATH="$OUTPUT_PATH/$EXPERIMENT_NAME/$MODEL_PATH_LAST"
    LOG_FILE="$CURRENT_OUTPUT_PATH/${TASK_NAME}.log"
    PORT=$((BASE_PORT + i))

    mkdir -p "$CURRENT_OUTPUT_PATH"

    {
        echo "Task: $TASK_NAME"
        echo "Output path: $CURRENT_OUTPUT_PATH"
        echo "Generation parameters: $GEN_KWARGS"
        echo "Model: llava_dream"
        echo "Conversation template: dream"
        echo "Model path: $MODEL_PATH"
        echo "GPU IDs: $CUDA_VISIBLE_DEVICES"
        echo "Processes: $NUM_PROCESSES"
        echo "Main process port: $PORT"
        echo "Task number: $((i + 1)) / $TOTAL_TASKS"
        echo "----------------------------------------"
    } > "$LOG_FILE"

    echo "Starting task $((i + 1))/$TOTAL_TASKS: $MODEL_PATH_LAST / $TASK_NAME"

    PYTHONUNBUFFERED=1 accelerate launch \
        --main_process_port "$PORT" \
        --num_processes "$NUM_PROCESSES" \
        -m lmms_eval \
        --model llava_dream \
        --gen_kwargs "$GEN_KWARGS" \
        --model_args "pretrained=$MODEL_PATH,conv_template=dream,model_name=llava_dream,attn_implementation=sdpa,device_map=auto" \
        --tasks "$TASK_NAME" \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix "$TASK_NAME" \
        --output_path "$CURRENT_OUTPUT_PATH" >> "$LOG_FILE" 2>&1

    status=$?
    if (( status != 0 )); then
        echo "Task $TASK_NAME failed with exit code $status" | tee -a "$LOG_FILE"
        ((FAILED_TASKS += 1))
    else
        echo "Task $TASK_NAME completed." | tee -a "$LOG_FILE"
    fi
done

if (( FAILED_TASKS > 0 )); then
    echo "$FAILED_TASKS of $TOTAL_TASKS evaluation task(s) failed." >&2
    exit 1
fi

echo "All $TOTAL_TASKS evaluation task(s) completed successfully."
