#!/bin/bash

# --- Tùy chọn: Copy checkpoint thủ công nếu cần resume từ một checkpoint cụ thể không có trong output_dir ---
# Ví dụ:
# CKPT_SRC_DIR_ON_KAGGLE=/kaggle/input/my-previous-checkpoint/checkpoint-epoch-5 
# TARGET_CKPT_DIR=./model/code2review_t5_data_task2/outputs/checkpoint-epoch-5
# if [ -d "$CKPT_SRC_DIR_ON_KAGGLE" ]; then
#   echo "Copying checkpoint from $CKPT_SRC_DIR_ON_KAGGLE to $TARGET_CKPT_DIR"
#   mkdir -p ${TARGET_CKPT_DIR}
#   cp ${CKPT_SRC_DIR_ON_KAGGLE}/checkpoint.pt ${TARGET_CKPT_DIR}/
# else
#   echo "Source checkpoint $CKPT_SRC_DIR_ON_KAGGLE not found. Will attempt to resume from existing output_dir if any."
# fi
# --- Kết thúc phần copy checkpoint tùy chọn ---


# Cài đặt/Nâng cấp thư viện (nếu cần, ví dụ cho QLoRA)
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # ví dụ cho CUDA 11.8
pip install transformers datasets sentencepiece accelerate bitsandbytes evaluate rouge_score nltk tensorboard # Các thư viện phổ biến

export TOKENIZERS_PARALLELISM=false

# Tạo các thư mục
OUTPUT_BASE_DIR=./model/code2review_t5_qlora_task2 # Đổi tên thư mục để phân biệt với lần chạy không QLoRA
mkdir -p ${OUTPUT_BASE_DIR}/cache
mkdir -p ${OUTPUT_BASE_DIR}/outputs/results # outputs/results
mkdir -p ${OUTPUT_BASE_DIR}/summary

# Tham số
TASK_NAME="refine"
SUB_TASK="small"
MODEL_TYPE="codet5"
MODEL_NAME="Salesforce/codet5-base" # Hoặc Salesforce/codet5-large nếu dùng large
DATA_DIR="/kaggle/input/daataa5/task2_data/t5_data/codet5_format_data" # Cập nhật đường dẫn data của bạn
OUTPUT_DIR="${OUTPUT_BASE_DIR}/outputs"
CACHE_PATH="${OUTPUT_BASE_DIR}/cache"
SUMMARY_DIR="${OUTPUT_BASE_DIR}/summary"
RES_DIR="${OUTPUT_DIR}/results"
RES_FN="${RES_DIR}/summary_codet5_qlora.txt"

NUM_TRAIN_EPOCHS=10 # Ví dụ: 5 epochs
LEARNING_RATE=1e-5 # Giữ nguyên hoặc điều chỉnh
WARMUP_STEPS=100 # Giảm nếu dataset nhỏ hoặc epoch ít
GRAD_ACC_STEPS=2
TRAIN_BATCH_SIZE=8 # Điều chỉnh dựa trên VRAM
EVAL_BATCH_SIZE=8
MAX_SOURCE_LENGTH=512
MAX_TARGET_LENGTH=100
BEAM_SIZE=5
PATIENCE_VAL=3 # Patience cho early stopping
LOGGING_STEPS_VAL=50 # Log mỗi 50 global steps

# Chạy script huấn luyện CodeT5 với QLoRA và resume
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
    --logging_steps ${LOGGING_STEPS_VAL} \
    # --save_last_checkpoints --always_save_model # Các cờ này có thể không cần thiết nữa
    #                                            # vì đã có lưu mỗi epoch và best checkpoint.
    #                                            # Tuy nhiên, chúng không gây hại nếu args không dùng đến.
    # --do_test # Thêm cờ này nếu bạn muốn chạy test sau khi huấn luyện xong

echo "Training finished. Output at ${OUTPUT_DIR}"
