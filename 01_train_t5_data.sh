#!/bin/bash

# --- Tùy chọn: Copy checkpoint thủ công nếu cần resume ---
# ... (giữ nguyên nếu bạn cần) ...

# Cài đặt thư viện
echo "INFO: Updating pip..."
pip install --upgrade pip

echo "INFO: Uninstalling potentially conflicting torch/torchvision versions..."
pip uninstall torch torchvision torchaudio -y

echo "INFO: Installing PyTorch, torchvision, torchaudio for a specific CUDA version (e.g., cu118 or cu121/cu124)"
# Lựa chọn 1: Cài cho CUDA 11.8 (thường ổn định cho nhiều thư viện)
# Hãy chắc chắn GPU của bạn hỗ trợ CUDA 11.8 runtime.
# pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Lựa chọn 2: Cài cho CUDA 12.1 (Nếu GPU hỗ trợ tốt và bitsandbytes tương thích)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Lựa chọn 3: Dựa theo phiên bản PyTorch mà Kaggle có thể đã cài sẵn (ví dụ cu124 từ log của bạn)
# Nếu bạn muốn giữ phiên bản PyTorch mặc định của Kaggle, hãy đảm bảo torchvision/torchaudio khớp.
# Thông thường, nếu bạn không chỉ định phiên bản CUDA, pip sẽ cố gắng cài phiên bản mới nhất phù hợp.
# Tuy nhiên, để kiểm soát, hãy thử chỉ định rõ. Ví dụ với cu121 (phổ biến hơn cu124 một chút cho các gói):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# HOẶC nếu bạn chắc chắn muốn cu118 cho bitsandbytes và các gói khác:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


echo "INFO: Installing other dependencies..."
pip install transformers datasets sentencepiece accelerate bitsandbytes evaluate rouge_score nltk tensorboard # Các thư viện phổ biến

export TOKENIZERS_PARALLELISM=false

# ... (phần còn lại của script train.sh của bạn giữ nguyên) ...

# Tạo các thư mục
OUTPUT_BASE_DIR=./model/code2review_t5_qlora_task2 
mkdir -p ${OUTPUT_BASE_DIR}/cache
mkdir -p ${OUTPUT_BASE_DIR}/outputs/results 
mkdir -p ${OUTPUT_BASE_DIR}/summary

# Tham số (giữ nguyên như bạn đã có)
TASK_NAME="refine"
SUB_TASK="small"
MODEL_TYPE="codet5"
MODEL_NAME="Salesforce/codet5-base" 
DATA_DIR="/kaggle/input/daataa10/task2_data/t5_data/codet5_format_data" 
OUTPUT_DIR="${OUTPUT_BASE_DIR}/outputs"
CACHE_PATH="${OUTPUT_BASE_DIR}/cache"
SUMMARY_DIR="${OUTPUT_BASE_DIR}/summary"
RES_DIR="${OUTPUT_DIR}/results"
RES_FN="${RES_DIR}/summary_codet5_qlora.txt"

NUM_TRAIN_EPOCHS=5 
LEARNING_RATE=1e-5 
WARMUP_STEPS=100 
GRAD_ACC_STEPS=2
TRAIN_BATCH_SIZE=8 
EVAL_BATCH_SIZE=8
MAX_SOURCE_LENGTH=512
MAX_TARGET_LENGTH=100
BEAM_SIZE=5
PATIENCE_VAL=3 
LOGGING_STEPS_VAL=50 

# Chạy script huấn luyện
echo "INFO: Starting training script..."
CUDA_VISIBLE_DEVICES=0 python run_gen.py \
    --do_train \
    --do_eval \
    --do_eval_bleu \
    --task ${TASK_NAME} \
    --sub_task ${SUB_TASK} \
    --model_type ${MODEL_TYPE} \
    --data_num -1 \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --warmup_steps ${WARMUP_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --patience ${PATIENCE_VAL} \
    --beam_size ${BEAM_SIZE} \
    --max_grad_norm 1.0 \
    --gradient_accumulation_steps ${GRAD_ACC_STEPS} \
    --tokenizer_name=${MODEL_NAME} \
    --model_name_or_path=${MODEL_NAME} \
    --use_lora \
    --bits 4 \
    --quant_type nf4 \
    --fp16 \
    --data_dir "${DATA_DIR}" \
    --cache_path ${CACHE_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --summary_dir ${SUMMARY_DIR} \
    --res_dir ${RES_DIR} \
    --res_fn ${RES_FN} \
    --train_batch_size ${TRAIN_BATCH_SIZE} \
    --eval_batch_size ${EVAL_BATCH_SIZE} \
    --max_source_length ${MAX_SOURCE_LENGTH} \
    --max_target_length ${MAX_TARGET_LENGTH} \
    --resume_from_checkpoint \
    --logging_steps ${LOGGING_STEPS_VAL}

echo "Training finished. Output at ${OUTPUT_DIR}"
