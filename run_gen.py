# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import argparse
import math
import numpy as np
from tqdm import tqdm
import multiprocessing
import time
import glob # For finding checkpoint files

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup # scheduler
from models import build_or_load_gen_model
from evaluator import smooth_bleu # Assuming smooth_bleu is available for certain tasks
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu # Assuming _bleu can take file paths or lists of strings
from utils import get_filenames, get_elapse_time, load_and_cache_gen_data
from configs import add_args, set_seed, set_dist # set_seed, set_dist

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def find_latest_epoch_checkpoint(output_dir):
    """Finds the latest epoch checkpoint file based on modification time."""
    potential_ckpt_dirs = glob.glob(os.path.join(output_dir, 'checkpoint-epoch-*'))
    ckpt_files = []
    for d in potential_ckpt_dirs:
        ckpt_file = os.path.join(d, 'checkpoint.pt')
        if os.path.exists(ckpt_file):
            ckpt_files.append(ckpt_file)
    
    if not ckpt_files:
        return None

    latest_ckpt_file = max(ckpt_files, key=os.path.getctime)
    logger.info(f"Found latest epoch checkpoint file: {latest_ckpt_file}")
    return latest_ckpt_file


# --- eval_ppl_epoch ---
def eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=args.cpu_cont if hasattr(args, 'cpu_cont') else 4, # use cpu_cont if available
                                 pin_memory=True)
    logger.info("  " + "***** Running PPL evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    eval_loss, batch_num = 0, 0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval PPL", disable=args.local_rank not in [-1, 0]):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, target_ids = batch
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        target_mask = target_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            # Adjust for model type if necessary, but codet5 uses this structure
            outputs = model(input_ids=source_ids, attention_mask=source_mask,
                            labels=target_ids, decoder_attention_mask=target_mask)
            loss = outputs.loss
        
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel): # if using DDP, loss is already mean
            pass # loss is already averaged by DDP
        elif args.n_gpu > 1 : # For DataParallel
             loss = loss.mean()
        eval_loss += loss.item()
        batch_num += 1
    
    if batch_num == 0: return float('inf') # Avoid division by zero
    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)
    return eval_ppl

# --- eval_bleu_epoch ---
def eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria):
    logger.info("  ***** Running BLEU evaluation on {} data ({}) *****".format(split_tag, criteria))
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_data)
    
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=args.cpu_cont if hasattr(args, 'cpu_cont') else 4,
                                 pin_memory=True)

    model.eval()
    pred_ids = []
    # For tasks like summarization, early_stopping might be true. For others, false.
    # Defaulting to True as it's common for generation. Make it an arg if needed.
    early_stopping_generation = getattr(args, 'early_stopping_generation', True)

    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc=f"Eval BLEU for {split_tag} set", disable=args.local_rank not in [-1, 0]):
        # If only_src=True (common for test), batch will be a tuple of one tensor
        source_ids = batch[0].to(args.device) 
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            # Use generate method of the underlying model if DataParallel or DDP wraps it
            model_to_generate = model.module if hasattr(model, 'module') else model
            preds = model_to_generate.generate(source_ids,
                                   attention_mask=source_mask,
                                   use_cache=True,
                                   num_beams=args.beam_size,
                                   early_stopping=early_stopping_generation,
                                   max_length=args.max_target_length)
            pred_ids.extend(list(preds.cpu().numpy()))

    pred_nls = [tokenizer.decode(id_seq, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id_seq in pred_ids]

    os.makedirs(args.res_dir, exist_ok=True)
    
    output_fn = os.path.join(args.res_dir, "{}_{}.output".format(split_tag, criteria))
    gold_fn = os.path.join(args.res_dir, "{}_{}.gold".format(split_tag, criteria)) 
    src_fn = os.path.join(args.res_dir, "{}_{}.src".format(split_tag, criteria))  

    golds = []
    sources = []
    # `eval_examples` are the raw InputFeatures/Examples loaded by `load_and_cache_gen_data`
    if eval_examples and hasattr(eval_examples[0], 'target') and eval_examples[0].target is not None:
         golds = [ex.target.strip() for ex in eval_examples]
    elif split_tag == 'test': # For test set, try to load gold from a predefined file if not in eval_examples
        # Example: try to load from args.test_filename by replacing extension or from a specific path
        # This part needs to be customized based on how your test gold files are stored.
        # For now, if not in eval_examples, golds will be empty for test set.
        logger.warning(f"Gold targets for test set ({split_tag}) not found directly in eval_examples. BLEU/EM will be 0 if not loaded externally.")


    if eval_examples and hasattr(eval_examples[0], 'source'):
         sources = [ex.source.strip() for ex in eval_examples]

    with open(output_fn, 'w', encoding='utf-8') as f_out, \
         open(gold_fn, 'w', encoding='utf-8') as f_gold, \
         open(src_fn, 'w', encoding='utf-8') as f_src:
        for i, pred_nl in enumerate(pred_nls):
            f_out.write(pred_nl.strip() + '\n')
            if i < len(golds):
                f_gold.write(golds[i] + '\n')
            if i < len(sources):
                f_src.write(sources[i] + '\n')
    logger.info("Saved predictions to %s", output_fn)
    if golds: logger.info("Saved gold references to %s", gold_fn)
    if sources: logger.info("Saved sources to %s", src_fn)

    bleu_score = 0.0
    em_score = 0.0
    code_bleu_score_val = 0.0

    if golds and pred_nls: # Calculate BLEU and EM only if gold references are available
        # Exact Match
        dev_accs = [pred_nls[i].strip() == golds[i] for i in range(min(len(pred_nls), len(golds)))] # Ensure lists are same length for comparison
        em_score = np.mean(dev_accs) * 100 if dev_accs else 0.0

        # BLEU score
        try:
            # Assuming _bleu from evaluator.bleu handles file paths
            bleu_score = round(_bleu(gold_fn, output_fn), 2) 
        except Exception as e:
            logger.error(f"Could not calculate BLEU score using _bleu(gold_fn, output_fn): {e}")
            bleu_score = 0.0
        
        # CodeBLEU (if applicable for the task)
        # lang = getattr(args, 'lang', 'java') # Default to java if lang not in args
        # if getattr(args, 'do_eval_codebleu', False): # Add a flag if you want to control this
        #     try:
        #         # Ensure calc_code_bleu arguments are correct (refs, hyp, lang, weights, tokenizer)
        #         # This might require specific formatting for reference files if multiple references.
        #         # Example: if gold_fn contains multiple references per line or needs specific processing.
        #         # For a single reference file, gold_fn is usually okay.
        #         code_bleu_results = calc_code_bleu.calc_code_bleu(refs=[gold_fn], hyp=output_fn, lang=lang, params='0.25,0.25,0.25,0.25')
        #         code_bleu_score_val = round(code_bleu_results['codebleu'] * 100, 2)
        #     except Exception as e:
        #         logger.warning(f"Could not calculate CodeBLEU: {e}")
        #         code_bleu_score_val = 0.0
    else:
        logger.warning("Gold references or predictions are empty. BLEU/EM scores will be 0.")


    result = {'em': em_score, 'bleu': bleu_score} # 'codebleu': code_bleu_score_val
    logger.info("***** Eval results for {} ({}) *****".format(split_tag, criteria))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))
    return result


def main():
    parser = argparse.ArgumentParser()
    # Call add_args from configs.py
    # The parser object is modified in place by add_args, or add_args returns the modified parser
    # Let's assume add_args modifies in place or returns it
    updated_parser = add_args(parser) # Get args from configs.py
    
    # Add specific arguments for this script if not in configs.py
    updated_parser.add_argument("--resume_from_checkpoint", action='store_true',
                        help="Resume training from the latest available epoch checkpoint.")
    updated_parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log every X updates steps.")
    # Add patience if not already in configs.add_args
    if not any(action.dest == 'patience' for action in updated_parser._actions):
        updated_parser.add_argument("--patience", type=int, default=0,
                            help="Patience for early stopping (0 to disable).")


    args = updated_parser.parse_args()

    logger.info(args)
    t0 = time.time()

    use_amp = args.fp16 and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        logger.info(">>> Mixed Precision Training: FP16 ENABLED")

    set_dist(args) # Setup for distributed training if args.local_rank != -1
    set_seed(args) # Set seed for reproducibility

    # Ensure output_dir, summary_dir, cache_path, res_dir exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.summary_dir, exist_ok=True)
    os.makedirs(args.cache_path, exist_ok=True)
    os.makedirs(args.res_dir, exist_ok=True)

    config, model, tokenizer = build_or_load_gen_model(args)
    model.to(args.device)

    # Handle DataParallel or DistributedDataParallel
    if args.local_rank != -1: # DDP
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    elif args.n_gpu > 1: # DataParallel
        model = torch.nn.DataParallel(model)

    # Multiprocessing pool for data loading
    # args.cpu_cont is expected from configs.py
    pool = multiprocessing.Pool(args.cpu_cont if hasattr(args, 'cpu_cont') else os.cpu_count())
    
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(
        args.data_dir, args.task, args.sub_task)
    
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train:
        tb_writer = None
        if args.local_rank in [-1, 0]: # Only master process writes to TensorBoard
            # Ensure summary_dir path for TensorBoard is valid
            summary_log_dir = os.path.join(args.summary_dir, '/'.join(args.output_dir.split('/')[1:])) if '/' in args.output_dir else args.summary_dir
            os.makedirs(os.path.dirname(summary_log_dir), exist_ok=True)
            tb_writer = SummaryWriter(log_dir=summary_log_dir)

        start_epoch = 0
        global_step = 0
        best_ppl = float('inf')
        best_bleu_em = -1.0 # Using a single metric for best BLEU/EM for simplicity

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        
        # Placeholder for total_steps, will be calculated after DataLoader
        # Or loaded from checkpoint
        scheduler = None 

        if args.resume_from_checkpoint:
            latest_checkpoint_file = find_latest_epoch_checkpoint(args.output_dir)
            if latest_checkpoint_file:
                logger.info(f"Resuming training from checkpoint: {latest_checkpoint_file}")
                try:
                    checkpoint = torch.load(latest_checkpoint_file, map_location=args.device)
                    
                    model_to_load = model.module if hasattr(model, 'module') else model
                    model_to_load.load_state_dict(checkpoint['model_state_dict'])
                    
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    # Scheduler state will be loaded after scheduler is created
                    
                    if use_amp and 'scaler_state_dict' in checkpoint and scaler:
                        scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    
                    start_epoch = checkpoint.get('epoch', -1) + 1
                    global_step = checkpoint.get('global_step', 0)
                    best_ppl = checkpoint.get('best_ppl', float('inf'))
                    best_bleu_em = checkpoint.get('best_bleu_em', -1.0)
                    
                    # Store scheduler state to load after scheduler is initialized
                    scheduler_state_to_load = checkpoint.get('scheduler_state_dict')

                    logger.info(f"Successfully resumed. Next epoch: {start_epoch}. Global step: {global_step}")
                    logger.info(f"Resumed best_ppl: {best_ppl:.4f}, best_bleu_em: {best_bleu_em:.4f}")

                except Exception as e:
                    logger.error(f"Failed to load checkpoint {latest_checkpoint_file}. Error: {e}", exc_info=True)
                    logger.info("Starting training from scratch.")
                    start_epoch = 0; global_step = 0; best_ppl = float('inf'); best_bleu_em = -1.0
                    scheduler_state_to_load = None
            else:
                logger.info("No 'checkpoint-epoch-*/checkpoint.pt' found for resuming. Starting training from scratch.")
                scheduler_state_to_load = None
        else:
            logger.info("Not resuming from checkpoint. Starting training from scratch.")
            scheduler_state_to_load = None

        train_examples, train_data = load_and_cache_gen_data(
            args, args.train_filename, pool, tokenizer, 'train')
        
        train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data, shuffle=True)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=args.train_batch_size,
            num_workers=args.cpu_cont if hasattr(args, 'cpu_cont') else 4,
            pin_memory=True,
            drop_last=True # Good for DDP to ensure all ranks have same number of batches
        )

        # Calculate total optimization steps
        # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) # More robust
        num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        if num_update_steps_per_epoch == 0 : num_update_steps_per_epoch = 1 # Handle small datasets

        t_total = num_update_steps_per_epoch * args.num_train_epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
        if scheduler_state_to_load:
            scheduler.load_state_dict(scheduler_state_to_load)
            logger.info("Loaded scheduler state.")


        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
        effective_batch_size = args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", effective_batch_size)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Starting epoch = %d", start_epoch)
        logger.info("  Starting global step = %d", global_step)

        dev_dataset = {}
        not_loss_dec_cnt = 0
        not_bleu_em_inc_cnt = 0 if args.do_eval_bleu else float('inf')

        model.zero_grad() # Ensure grads are zeroed before starting training loop
        for epoch in range(start_epoch, args.num_train_epochs):
            if args.local_rank != -1: # For DDP, set epoch for sampler
                train_dataloader.sampler.set_epoch(epoch)

            logger.info(f"Epoch {epoch}/{args.num_train_epochs - 1}")
            bar = tqdm(train_dataloader, desc=f"Epoch {epoch} Training", disable=args.local_rank not in [-1, 0])
            
            model.train()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(args.device) for t in batch)
                source_ids, target_ids = batch
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                target_mask = target_ids.ne(tokenizer.pad_token_id)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(
                        input_ids=source_ids, attention_mask=source_mask,
                        labels=target_ids, decoder_attention_mask=target_mask
                    )
                    loss = outputs.loss

                if args.n_gpu > 1 and not isinstance(model, DDP): # For DataParallel
                    loss = loss.mean() 
                # For DDP, loss is already averaged across GPUs if reduction='mean' (default for CrossEntropyLoss)
                
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                current_loss_val = loss.item() # For logging

                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad() # model.zero_grad() is also fine
                    global_step += 1

                    if args.local_rank in [-1, 0] and tb_writer and global_step % args.logging_steps == 0:
                        tb_writer.add_scalar('train/loss', current_loss_val, global_step)
                        tb_writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
                
                if args.local_rank in [-1, 0]:
                     bar.set_postfix(loss=f"{current_loss_val:.4f}", lr=f"{scheduler.get_last_lr()[0]:.1e}", gs=global_step)
            
            if args.local_rank in [-1, 0]: # Master process saves checkpoints and evaluates
                epoch_checkpoint_dir = os.path.join(args.output_dir, f'checkpoint-epoch-{epoch}')
                os.makedirs(epoch_checkpoint_dir, exist_ok=True)
                model_to_save = model.module if hasattr(model, 'module') else model
                
                current_checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'global_step': global_step,
                    'best_ppl': best_ppl,
                    'best_bleu_em': best_bleu_em,
                    'args_dict': vars(args), # Save args as dict for easier loading
                }
                if use_amp and scaler:
                    current_checkpoint_data['scaler_state_dict'] = scaler.state_dict()
                
                torch.save(current_checkpoint_data, os.path.join(epoch_checkpoint_dir, 'checkpoint.pt'))
                logger.info(f"Saved epoch {epoch} checkpoint to {os.path.join(epoch_checkpoint_dir, 'checkpoint.pt')}")

                if args.do_eval:
                    if 'dev_loss' not in dev_dataset:
                        eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev')
                        dev_dataset['dev_loss'] = (eval_examples, eval_data)
                    else:
                        eval_examples, eval_data = dev_dataset['dev_loss']
                    
                    eval_ppl = eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer)
                    logger.info(f"Epoch {epoch} | Global Step {global_step} | Eval PPL: {eval_ppl:.4f}")
                    if tb_writer: tb_writer.add_scalar('eval/ppl', eval_ppl, global_step)
                    fa.write(f"Epoch {epoch} | GS {global_step} | Eval PPL: {eval_ppl:.4f}\n")

                    if eval_ppl < best_ppl:
                        not_loss_dec_cnt = 0
                        logger.info(f"  Best PPL improved from {best_ppl:.4f} to {eval_ppl:.4f}")
                        best_ppl = eval_ppl
                        fa.write(f"  Best PPL: {best_ppl:.4f} (Epoch {epoch})\n")
                        
                        best_ppl_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                        os.makedirs(best_ppl_dir, exist_ok=True)
                        # Update best_ppl in the checkpoint data before saving
                        current_checkpoint_data_for_best = current_checkpoint_data.copy()
                        current_checkpoint_data_for_best['best_ppl'] = best_ppl 
                        torch.save(current_checkpoint_data_for_best, os.path.join(best_ppl_dir, 'checkpoint.pt'))
                        logger.info(f"Saved best-PPL chkpt (Epoch {epoch}) to {best_ppl_dir}/checkpoint.pt")
                    else:
                        not_loss_dec_cnt += 1
                        logger.info(f"  PPL not improved for {not_loss_dec_cnt} epoch(s). Best PPL: {best_ppl:.4f}")
                    
                    if args.do_eval_bleu:
                        # Re-use eval_examples and eval_data if suitable for BLEU
                        # Note: load_and_cache_gen_data for 'dev' should provide targets in eval_examples.
                        bleu_results = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'dev', f'e{epoch}_gs{global_step}')
                        # Use a combined metric or just BLEU for best tracking, e.g., BLEU
                        current_eval_metric = bleu_results['bleu'] 
                        if tb_writer: 
                            tb_writer.add_scalar('eval/bleu', bleu_results['bleu'], global_step)
                            tb_writer.add_scalar('eval/em', bleu_results['em'], global_step)
                            # if 'codebleu' in bleu_results: tb_writer.add_scalar('eval/codebleu', bleu_results['codebleu'], global_step)
                        fa.write(f"Epoch {epoch} | Eval BLEU: {bleu_results['bleu']:.2f} | Eval EM: {bleu_results['em']:.2f}\n")

                        if current_eval_metric > best_bleu_em:
                            not_bleu_em_inc_cnt = 0
                            logger.info(f"  Best BLEU/EM metric improved from {best_bleu_em:.2f} to {current_eval_metric:.2f}")
                            best_bleu_em = current_eval_metric
                            fa.write(f"  Best BLEU/EM: {best_bleu_em:.2f} (Epoch {epoch})\n")

                            best_bleu_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                            os.makedirs(best_bleu_dir, exist_ok=True)
                            current_checkpoint_data_for_best = current_checkpoint_data.copy()
                            current_checkpoint_data_for_best['best_bleu_em'] = best_bleu_em
                            torch.save(current_checkpoint_data_for_best, os.path.join(best_bleu_dir, 'checkpoint.pt'))
                            logger.info(f"Saved best-BLEU chkpt (Epoch {epoch}) to {best_bleu_dir}/checkpoint.pt")
                        else:
                            not_bleu_em_inc_cnt +=1
                            logger.info(f"  BLEU/EM not improved for {not_bleu_em_inc_cnt} epoch(s). Best: {best_bleu_em:.2f}")
                    
                    if args.patience > 0 and not_loss_dec_cnt >= args.patience and not_bleu_em_inc_cnt >= args.patience:
                        logger.info(f"Early stopping triggered at epoch {epoch} (PPL no improvement for {not_loss_dec_cnt}, BLEU/EM for {not_bleu_em_inc_cnt} epochs).")
                        fa.write(f"Early stopping at epoch {epoch}.\n")
                        break 
            
            if args.local_rank != -1: # Sync distributed processes if any
                torch.distributed.barrier()
        # End of epoch loop

        if args.local_rank in [-1, 0] and tb_writer:
            tb_writer.close()
        logger.info("Training done in %s", get_elapse_time(t0))
        fa.write(f"Training done in {get_elapse_time(t0)}\n")

    if args.do_test and args.local_rank in [-1, 0]:
        logger.info("***** Testing *****")
        test_criteria_names = ['best-ppl', 'best-bleu']
        # Optionally, find and add the very last epoch checkpoint
        # latest_epoch_ckpt_file = find_latest_epoch_checkpoint(args.output_dir)
        # if latest_epoch_ckpt_file:
        #     last_epoch_dir_name = os.path.basename(os.path.dirname(latest_epoch_ckpt_file))
        #     if last_epoch_dir_name not in test_criteria_names:
        #         test_criteria_names.append(last_epoch_dir_name)


        for criteria_name in test_criteria_names:
            checkpoint_file_to_test = os.path.join(args.output_dir, criteria_name, 'checkpoint.pt')

            if not os.path.exists(checkpoint_file_to_test):
                logger.warning(f"Checkpoint file {checkpoint_file_to_test} for criteria '{criteria_name}' not found. Skipping test.")
                continue

            logger.info(f"Loading model for testing from: {checkpoint_file_to_test}")
            try:
                checkpoint = torch.load(checkpoint_file_to_test, map_location=args.device)
                model_to_load = model.module if hasattr(model, 'module') else model
                
                if 'model_state_dict' in checkpoint:
                    model_to_load.load_state_dict(checkpoint['model_state_dict'])
                else: # For legacy checkpoints that only contain state_dict
                    model_to_load.load_state_dict(checkpoint)
                model.eval()
            except Exception as e:
                logger.error(f"Failed to load model from {checkpoint_file_to_test} for testing: {e}", exc_info=True)
                continue

            test_examples, test_data = load_and_cache_gen_data(
                args, args.test_filename, pool, tokenizer, 'test', only_src=True # only_src=True for test
            )
            
            # `eval_bleu_epoch` needs to handle loading gold references for test set internally if not in `test_examples`
            test_results = eval_bleu_epoch(args, test_data, test_examples, model, tokenizer, 'test', criteria_name)
            result_str = f"Criteria: {criteria_name} | Test BLEU: {test_results['bleu']:.2f} | Test EM: {test_results['em']:.2f}"
            # if 'codebleu' in test_results: result_str += f" | Test CodeBLEU: {test_results['codebleu']:.2f}"
            logger.info(result_str)
            fa.write(result_str + "\n")
            if args.res_fn: # Save to detailed results file
                with open(args.res_fn, 'a+', encoding='utf-8') as f_res:
                    f_res.write(f"[Time: {get_elapse_time(t0)}] Checkpoint: {checkpoint_file_to_test}\n")
                    f_res.write(result_str + "\n")
    
    logger.info("Process finished in %s", get_elapse_time(t0))
    fa.write(f"Process finished in {get_elapse_time(t0)}\n")
    fa.close()
    if pool:
        pool.close()
        pool.join()
    
    if args.local_rank != -1:
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
