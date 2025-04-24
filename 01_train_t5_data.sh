# 0) Gỡ các build PyTorch/Transformers cũ
pip uninstall -y torch torchvision torchaudio transformers huggingface-hub fsspec bitsandbytes

# 1) Cài PyTorch và các extension khớp với CUDA 11.8 (được driver CUDA 12.6 hỗ trợ ngược)
pip install --no-cache-dir \
  torch==2.1.0+cu118 \
  torchvision==0.16.0+cu118 \
  torchaudio==2.1.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

# 2) Cài quantization engine bitsandbytes cho CUDA 11.8
pip install --no-cache-dir bitsandbytes-cuda118

# 3) Hạ transformers về 4.34.0 (loại bỏ torchao dependency)
#    và cài huggingface-hub >=0.25 & fsspec <=2024.12 để hài hoà với peft/datasets
pip install --no-cache-dir \
  transformers==4.34.0 \
  huggingface-hub==0.25.3 \
  fsspec==2024.12.0

# 4) Thiết đường dẫn CUDA (nếu cần)
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda

# 5) Tạo các thư mục
mkdir -p model
mkdir -p ./model/code2review_t5_data_task2/{cache,outputs,summary,outputs/results}

# Chạy script huấn luyện CodeT5 với QLoRA
CUDA_VISIBLE_DEVICES=0 python run_gen.py --do_train --do_eval --do_eval_bleu \
        --task refine --sub_task small --model_type codet5 --data_num -1 \
        --num_train_epochs 3 \
        --warmup_steps 500 \
        --learning_rate 5e-5 --patience 3 --beam_size 5 \
        --gradient_accumulation_steps 1 \
        --tokenizer_name=Salesforce/codet5-base \
        --model_name_or_path=Salesforce/codet5-base \
        --use_lora \
        --bits 4 --double_quant --quant_type nf4 \
        --bf16 \
        --data_dir "/kaggle/input/daataa10/task2_data/t5_data/codet5_format_data" \
        --cache_path ./model/code2review_t5_data_task2/cache/ \
        --output_dir ./model/code2review_t5_data_task2/outputs/ \
        --summary_dir ./model/code2review_t5_data_task2/summary/ --save_last_checkpoints --always_save_model \
        --res_dir ./model/code2review_t5_data_task2/outputs/results \
        --res_fn ./model/code2review_t5_data_task2/outputs/results/summarize_codet5_base.txt \
        --train_batch_size 8 --eval_batch_size 8 --max_source_length 512 --max_target_length 100
