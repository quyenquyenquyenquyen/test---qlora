# 0) Gỡ bản PyTorch không tương thích
pip uninstall -y torch torchvision torchaudio

# 1) Cài lại PyTorch cho CUDA 11.7
pip install --no-cache-dir torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu117

# 2) Xác nhận import
python - <<'PYCODE'
import torch
print(torch.__version__, torch.version.cuda)
PYCODE

# 3) Nếu ổn, tiếp tục cài bitsandbytes tương thích
pip install --upgrade bitsandbytes-cuda117

# Tạo các thư mục cần thiết
mkdir -p model
mkdir -p ./model/code2review_t5_data_task2/
mkdir -p ./model/code2review_t5_data_task2/cache/
mkdir -p ./model/code2review_t5_data_task2/outputs/
mkdir -p ./model/code2review_t5_data_task2/summary/
mkdir -p ./model/code2review_t5_data_task2/outputs/results

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
