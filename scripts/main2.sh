#!/bin/bash  
  
# main2.sh - 4 GPU version of the R-Zero pipeline  
# This script orchestrates question generation, evaluation, and training on 4 GPUs  
  
export STORAGE_PATH="${STORAGE_PATH:-./storage}"  
mkdir -p "$STORAGE_PATH/generated_question"  
  
# Configuration  
MODEL_NAME="${1:-Qwen/Qwen3-4B-Base}"  
NUM_SAMPLES="${2:-100}"  
SAVE_NAME="${3:-questions_4gpu}"  
  
echo "=== R-Zero Pipeline (4 GPU Configuration) ==="
echo "Model: $MODEL_NAME"
echo "Samples per GPU: $NUM_SAMPLES"
echo "Save name: $SAVE_NAME"
echo "=============================================="
  
# Step 1: Question Generation (4 GPUs in parallel)  
echo "[Step 1] Starting question generation on 4 GPUs..."
bash question_generate/question_generate_4gpu.bash "$MODEL_NAME" "$NUM_SAMPLES" "$SAVE_NAME"  
  
# Step 2: Start vLLM evaluation server (single GPU with tensor parallelism)  
echo "[Step 2] Starting vLLM evaluation server..."  
python vllm_service_init/start_vllm_server_4gpu.py \
    --port 5000 \
    --model_path "$MODEL_NAME" \
    --gpu_mem_util 0.7 \
    --tensor_parallel_size 4 &
SERVER_PID=$!  
  
# Wait for server to be ready  
sleep 30  
  
# Step 3: Run evaluation  
echo "[Step 3] Running evaluation..."  
python evaluation/eval_bbeh_4gpu.py \
    --model_path "$MODEL_NAME" \
    --output_file "outputs_4gpu.json"  
  
# Cleanup  
echo "[Cleanup] Shutting down vLLM server..."  
kill $SERVER_PID  
  
echo "=== Pipeline Complete ==="