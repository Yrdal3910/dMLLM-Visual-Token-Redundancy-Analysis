#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$EVAL_ROOT"

export PYTHONPATH="$EVAL_ROOT:$EVAL_ROOT/lmms-eval${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

OUTPUT_PATH="${OUTPUT_PATH:-$EVAL_ROOT/exp/llava_v_eval}"
TASK_NAMES="${TASK_NAMES:-mme}"
BASE_PORT="${BASE_PORT:-29500}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"

MODEL_PATHS=(
    "GSAI-ML/LLaDA-V"
)

MODEL="llava_onevision_llada"
MODEL_NAME="llava_llada"
CONV_TEMPLATE="llava_llada"

if ! command -v accelerate >/dev/null 2>&1; then
    echo "Error: accelerate is not available in PATH." >&2
    exit 127
fi

if [[ ! -d "$EVAL_ROOT/llava" ]]; then
    echo "Error: $EVAL_ROOT/llava does not exist." >&2
    echo "Rename the selected llava_* implementation to llava before running." >&2
    exit 1
fi

IFS=',' read -r -a RAW_TASKS <<< "$TASK_NAMES"
TASKS=()
for task in "${RAW_TASKS[@]}"; do
    task="${task#"${task%%[![:space:]]*}"}"
    task="${task%"${task##*[![:space:]]}"}"
    [[ -n "$task" ]] && TASKS+=("$task")
done

if (( ${#TASKS[@]} == 0 )); then
    echo "Error: TASK_NAMES does not contain any valid task." >&2
    exit 1
fi

if [[ ! "$BASE_PORT" =~ ^[0-9]+$ ]] || (( BASE_PORT < 1 || BASE_PORT > 65535 )); then
    echo "Error: BASE_PORT must be an integer between 1 and 65535." >&2
    exit 1
fi

IFS=',' read -r -a VISIBLE_GPUS <<< "$CUDA_VISIBLE_DEVICES"
GPU_COUNT=${#VISIBLE_GPUS[@]}
TOTAL_TASKS=$(( ${#MODEL_PATHS[@]} * ${#TASKS[@]} ))

generation_kwargs_for_task() {
    case "$1" in
        mmmu_val|mmmu_pro_standard|mmstar|ai2d|seedbench|mmbench_en_dev|mmmu_pro_vision|muirbench|videomme|mlvu_dev|mme|realworldqa)
            printf '%s' '{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":2,"block_length":2,"gen_steps":2,"think_mode":"no_think"}'
            ;;
        chartqa)
            printf '%s' '{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":16,"block_length":16,"gen_steps":8,"stopping_criteria":["\n"],"think_mode":"no_think"}'
            ;;
        docvqa_val|infovqa_val)
            printf '%s' '{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":32,"block_length":32,"gen_steps":16,"think_mode":"no_think"}'
            ;;
        mathvista_testmini)
            printf '%s' '{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":96,"block_length":96,"gen_steps":48,"think_mode":"think"}'
            ;;
        mathverse_testmini_vision)
            printf '%s' '{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":64,"block_length":64,"gen_steps":32,"think_mode":"think"}'
            ;;
        detailcaps|video_dc499|short_test)
            printf '%s' '{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":128,"block_length":128,"gen_steps":64,"think_mode":"think"}'
            ;;
        *)
            printf '%s' '{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":2,"block_length":2,"gen_steps":2,"think_mode":"no_think"}'
            ;;
    esac
}

echo "Evaluation root: $EVAL_ROOT"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES ($GPU_COUNT devices)"
echo "Total evaluation tasks: $TOTAL_TASKS (run sequentially)"

task_index=0
failed_count=0
failed_tasks=()

for model_path in "${MODEL_PATHS[@]}"; do
    model_path_last="${model_path##*/}"
    current_output_path="$OUTPUT_PATH/$model_path_last"
    mkdir -p "$current_output_path"

    for task_name in "${TASKS[@]}"; do
        task_index=$((task_index + 1))
        port=$((BASE_PORT + task_index - 1))
        if (( port > 65535 )); then
            echo "Error: computed port $port exceeds 65535." >&2
            exit 1
        fi

        gen_kwargs="$(generation_kwargs_for_task "$task_name")"
        log_file="$current_output_path/${task_name}.log"

        {
            echo "Task: $task_name"
            echo "Output path: $current_output_path"
            echo "Generation parameters: $gen_kwargs"
            echo "Model: $MODEL_NAME"
            echo "Conversation template: $CONV_TEMPLATE"
            echo "Model path: $model_path"
            echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
            echo "Accelerate processes: 1 (device_map=auto)"
            echo "Task number: $task_index / $TOTAL_TASKS"
            echo "Main process port: $port"
            echo "----------------------------------------"
        } > "$log_file"

        echo "Starting task $task_index/$TOTAL_TASKS on $GPU_COUNT visible GPUs (port $port): $model_path_last / $task_name"

        export MASTER_PORT="$port"
        if PYTHONUNBUFFERED=1 accelerate launch \
            --main_process_port "$port" \
            --num_processes 8 \
            -m lmms_eval \
            --model "$MODEL" \
            --gen_kwargs "$gen_kwargs" \
            --model_args "pretrained=$model_path,conv_template=$CONV_TEMPLATE,model_name=$MODEL_NAME,attn_implementation=$ATTN_IMPLEMENTATION,device_map=auto" \
            --tasks "$task_name" \
            --batch_size 1 \
            --log_samples \
            --log_samples_suffix "$task_name" \
            --output_path "$current_output_path" >> "$log_file" 2>&1
        then
            echo "Task $task_name completed." | tee -a "$log_file"
        else
            status=$?
            failed_count=$((failed_count + 1))
            failed_tasks+=("$model_path_last/$task_name (exit $status)")
            echo "Task $task_name failed with exit code $status." | tee -a "$log_file"
        fi
    done
done

if (( failed_count > 0 )); then
    echo "$failed_count of $TOTAL_TASKS evaluation tasks failed:" >&2
    printf '  - %s\n' "${failed_tasks[@]}" >&2
    exit 1
fi

echo "All $TOTAL_TASKS evaluation tasks completed successfully."
