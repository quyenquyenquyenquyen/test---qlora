import random
import torch
import logging
import multiprocessing
import numpy as np

logger = logging.getLogger(__name__)

def add_args(parser):
    parser.add_argument("--task", type=str, required=True,
                        choices=['summarize', 'concode', 'translate', 'refine', 'defect', 'clone', 'multi_task'])
    parser.add_argument("--sub_task", type=str, default='')
    parser.add_argument("--lang", type=str, default='')
    parser.add_argument("--eval_task", type=str, default='')
    parser.add_argument("--model_type", default="codet5", type=str,
                        choices=['roberta', 'bart', 'codet5', 't5'])
    parser.add_argument("--add_lang_ids", action='store_true')
    parser.add_argument("--data_num", default=-1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--num_train_epochs", default=100, type=int)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--summary_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--res_dir", type=str, required=True)
    parser.add_argument("--res_fn", type=str, default='')
    parser.add_argument("--add_task_prefix", action='store_true',
                        help="Whether to add task prefix for t5 and codet5")
    parser.add_argument("--save_last_checkpoints", action='store_true')
    parser.add_argument("--always_save_model", action='store_true')
    parser.add_argument("--do_eval_bleu", action='store_true',
                        help="Whether to evaluate bleu on dev set.")

    # Required parameters
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str,
                        help="Pretrained model path or identifier")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="Directory for model predictions and checkpoints")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to existing model checkpoint (.bin file)")

    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA fine-tuning")
    parser.add_argument("--lora_r", default=8, type=int, help="LoRA rank")
    parser.add_argument("--lora_alpha", default=32, type=int, help="LoRA scaling alpha")
    parser.add_argument("--lora_dropout", default=0.1, type=float, help="Dropout for LoRA layers")

    # QLoRA / quantization arguments
    parser.add_argument("--bits", type=int, default=4, choices=[4,8,16,32],
                        help="Bits for quantization: 4 or 8")
    parser.add_argument("--double_quant", action="store_true",
                        help="Enable double quantization (bnb only)")
    parser.add_argument("--quant_type", type=str, default="nf4", choices=["fp4","nf4"],
                        help="4-bit quantization type")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 for compute")
    parser.add_argument("--fp16", action="store_true", help="Use float16 for compute")

    # I/O and training settings
    parser.add_argument("--train_filename", default=None, type=str,
                        help="Path to train .jsonl file")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="Path to dev .jsonl file")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="Path to test .jsonl file")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path")
    parser.add_argument("--tokenizer_name", default="roberta-base", type=str,
                        help="Tokenizer name or path")
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="Max source sequence length")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="Max target sequence length")

    parser.add_argument("--do_train", action='store_true', help="Run training")
    parser.add_argument("--do_eval", action='store_true', help="Run evaluation")
    parser.add_argument("--do_test", action='store_true', help="Run testing")
    parser.add_argument("--do_lower_case", action='store_true', help="Lowercase input text")
    parser.add_argument("--no_cuda", action='store_true', help="Disable CUDA")

    parser.add_argument("--train_batch_size", default=8, type=int, help="Training batch size")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Eval batch size")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="Learning rate")
    parser.add_argument("--beam_size", default=10, type=int, help="Beam size for generation")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Adam epsilon")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max grad norm")

    parser.add_argument("--save_steps", default=-1, type=int, help="Save checkpoint every N steps")
    parser.add_argument("--log_steps", default=-1, type=int, help="Log every N steps")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="Override num_train_epochs")
    parser.add_argument("--eval_steps", default=-1, type=int, help="Eval every N steps")
    parser.add_argument("--train_steps", default=-1, type=int, help="Training steps limit")
    parser.add_argument("--warmup_steps", default=100, type=int, help="Warmup steps")
    parser.add_argument("--local_rank", default=-1, type=int, help="Distributed local rank")
    parser.add_argument("--seed", default=1234, type=int, help="Random seed")

    args = parser.parse_args()

    # Task-specific defaults
    if args.task == 'summarize':
        args.lang = args.sub_task
    elif args.task in ['refine', 'concode', 'clone']:
        args.lang = 'java'
    elif args.task == 'defect':
        args.lang = 'c'
    elif args.task == 'translate':
        args.lang = 'c_sharp' if args.sub_task == 'java-cs' else 'java'
    return args

def set_dist(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        # Setup for distributed data parallel
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    cpu_cont = multiprocessing.cpu_count()
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, cpu count: %d",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), cpu_cont)
    args.device = device
    args.cpu_cont = cpu_cont


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
