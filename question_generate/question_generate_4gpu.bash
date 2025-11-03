#!/bin/bash  
  
# question_generate_4gpu.bash - 4 GPU parallel question generation  
  
model_name=$1  
num_samples=$2  
save_name=$3  

# Ensure STORAGE_PATH is set  
if [ -z "$STORAGE_PATH" ]; then
    echo "ERROR: STORAGE_PATH environment variable is not set"
    exit 1
fi

echo "Starting question generation with 4 GPUs..."  
echo "Model: $model_name"  
echo "Samples per GPU: $num_samples"  
echo "Save name: $save_name"  
  
# Launch 4 parallel processes, one per GPU  
for suffix in 0 1 2 3; do  
    echo "Launching GPU $suffix..."  
    CUDA_VISIBLE_DEVICES=$suffix python question_generate/question_generate.py \
        --model "$model_name" \
        --num_samples "$num_samples" \
        --save_name "$save_name" \
        --suffix "$suffix" &
done  
  
# Wait for all background processes to complete  
wait  
  
echo "Question generation complete. Output files:"  
ls -lh "${STORAGE_PATH}/generated_question/${save_name}_"*.json