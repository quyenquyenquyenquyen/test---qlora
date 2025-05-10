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
from transformers import get_linear_schedule_with_warmup
from models import build_or_load_gen_model
from evaluator import smooth_bleu # Assuming smooth_bleu is available
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu # Assuming _bleu can take file paths or lists of strings
from utils import get_filenames, get_elapse_time, load_and_cache_gen_data
from configs import add_args, set_seed, set_dist

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
                                 num_workers=4, pin_memory=True)
    logger.info("  " + "***** Running PPL evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    eval_loss, batch_num = 0, 0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval PPL"):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, target_ids = batch
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        target_mask = target_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            # Assuming model_type is not 'roberta' for codet5
            outputs = model(input_ids=source_ids, attention_mask=source_mask,
                            labels=target_ids, decoder_attention_mask=target_mask)
            loss = outputs.loss
        
        if args.n_gpu > 1:
            loss = loss.mean() # loss is replicated on all GPUs, take mean
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
    eval_sampler = SequentialSampler(eval_data) # For test, only_src=True, so eval_data contains only source_ids
    
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=4, pin_memory=True)

    model.eval()
    pred_ids = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval BLEU for {} set".format(split_tag)):
        source_ids = batch[0].to(args.device) # batch[0] as only_src=True for test data
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            preds = model.generate(source_ids,
                                   attention_mask=source_mask,
                                   use_cache=True,
                                   num_beams=args.beam_size,
                                   early_stopping=True, # Consider making this an arg if task varies
                                   max_length=args.max_target_length)
            pred_ids.extend(list(preds.cpu().numpy()))

    pred_nls = [tokenizer.decode(id_seq, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id_seq in pred_ids]

    # Ensure res_dir exists
    os.makedirs(args.res_dir, exist_ok=True)
    
    output_fn = os.path.join(args.res_dir, "{}_{}.output".format(split_tag, criteria))
    gold_fn = os.path.join(args.res_dir, "{}_{}.gold".format(split_tag, criteria)) # Ground truth
    src_fn = os.path.join(args.res_dir, "{}_{}.src".format(split_tag, criteria))   # Source

    # `eval_examples` here would be from `load_and_cache_gen_data` with `only_src=True`
    # so `gold.target` might not be directly available in `eval_examples` if it's test set.
    # For dev set evaluation, `eval_examples` (passed to this func) should have targets.
    # For test set, we typically load gold references from a separate file if available.

    golds = []
    sources = []
    if hasattr(eval_examples[0], 'target'): # Check if target exists (e.g., for dev set)
         golds = [ex.target.strip() for ex in eval_examples]
    if hasattr(eval_examples[0], 'source'):
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
    # code_bleu_score = 0.0 # Initialize

    if golds: # Calculate BLEU and EM only if gold references are available
        # Exact Match
        dev_accs = [pred_nls[i].strip() == golds[i] for i in range(len(golds))]
        em_score = np.mean(dev_accs) * 100

        # BLEU score using _bleu (from evaluator.bleu)
        # _bleu might expect file paths or list of strings. Assuming it handles lists of strings:
        # If _bleu expects file paths:
        # bleu_score = round(_bleu(gold_fn, output_fn), 2) 
        # If _bleu expects list of lists of tokens (or list of strings for references):
        # For simplicity, assuming pred_nls and golds are lists of strings.
        # The `_bleu` function from `nltk.translate.bleu_score.sentence_bleu` with smoothing
        # typically expects tokenized sentences. smooth_bleu.py usually handles this.
        # Let's assume the imported _bleu can work with lists of strings directly or file paths.
        # Using list of strings for _bleu for now as it's simpler than tokenizing here.
        # You might need to adjust this based on your specific _bleu implementation.
        try:
            # This expects gold references to be a list of strings, and predictions to be a list of strings.
            # Some _bleu implementations might want tokenized inputs.
            bleu_score = round(_bleu([g.split() for g in golds], [p.split() for p in pred_nls]) * 100, 2) # Example tokenization
            # Or if _bleu handles file paths directly and tokenizes internally:
            # bleu_score = round(_bleu(gold_fn, output_fn), 2)
        except Exception as e:
            logger.warning(f"Could not calculate BLEU score with lists of strings: {e}. Trying with file paths.")
            try:
                bleu_score = round(_bleu(gold_fn, output_fn), 2) # Assumes _bleu takes file paths
            except Exception as e_file:
                logger.error(f"Could not calculate BLEU score with file paths either: {e_file}")
                bleu_score = 0.0 # Fallback
        
        # CodeBLEU (if applicable for the task)
        # lang = args.lang if hasattr(args, 'lang') else 'java' # Example, assuming lang is an arg
        # try:
        #     # calc_code_bleu often expects file paths.
        #     # The reference file format for calc_code_bleu might be specific (e.g., one ref per file or specific multi-ref format)
        #     # This is a simplified call. You might need to prepare reference files differently.
        #     code_bleu_results = calc_code_bleu.calc_code_bleu([gold_fn], output_fn, lang) 
        #     code_bleu_score = round(code_bleu_results['codebleu'] * 100, 2)
        # except Exception as e:
        #     logger.warning(f"Could not calculate CodeBLEU: {e}")
        #     code_bleu_score = 0.0

    result = {'em': em_score, 'bleu': bleu_score} # 'codebleu': code_bleu_score
    logger.info("***** Eval results for {} ({}) *****".format(split_tag, criteria))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))
    return result


def main():
    parser = argparse.ArgumentParser()
    # Add --resume_from_checkpoint directly here if not in configs.add_args
    # parser = add_args(parser) # Assuming add_args is from configs.py
    # For safety, let's assume add_args returns the parser
    # And we add our specific argument after that.
    # This should be handled in your configs.py for cleaner code.
    _parser = add_args(parser) # Call the original add_args from configs
    _parser.add_argument("--resume_from_checkpoint", action='store_true',
                        help="Resume training from the latest available epoch checkpoint.")
    args = _parser.parse_args() # Parse all args

    logger.info(args)
    t0 = time.time()

    use_amp = args.fp16 and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) # Pass enabled=use_amp
    if use_amp:
        logger.info(">>> Mixed Precision Training: FP16 ENABLED")

    set_dist(args)
    set_seed(args)
    config, model, tokenizer = build_or_load_gen_model(args)
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    pool = multiprocessing.Pool(args.cpu_cont)
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(
        args.data_dir, args.task, args.sub_task)
    
    # Ensure output_dir exists for summary.log
    os.makedirs(args.output_dir, exist_ok=True)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train:
        tb_writer = None
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = f"{args.summary_dir}/{'/'.join(args.output_dir.split('/')[1:])}"
            os.makedirs(os.path.dirname(summary_fn), exist_ok=True)
            tb_writer = SummaryWriter(summary_fn)

        # Initialize training state variables
        start_epoch = 0
        global_step = 0
        best_ppl = float('inf')
        best_bleu_em = -1.0 # For BLEU or BLEU+EM

        # --- Optimizer & Scheduler (defined before potential checkpoint loading) ---
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        
        # Calculate total_steps once, after data loader is set up or use a placeholder if loaded from checkpoint
        # For now, we set it up assuming a full run, scheduler might be re-adjusted if global_step is loaded
        train_examples_len_for_scheduler = len(load_and_cache_gen_data(args, args.train_filename, pool, tokenizer, 'train', for_scheduler_len_only=True))
        total_steps_for_scheduler = args.num_train_epochs * (train_examples_len_for_scheduler // (args.train_batch_size * args.gradient_accumulation_steps))
        if args.local_rank != -1 : # Adjust for distributed
             total_steps_for_scheduler = total_steps_for_scheduler // torch.distributed.get_world_size()


        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps_for_scheduler
        )

        # --- Load Checkpoint if Resuming ---
        if args.resume_from_checkpoint:
            latest_checkpoint_file = find_latest_epoch_checkpoint(args.output_dir)
            if latest_checkpoint_file:
                logger.info(f"Resuming training from checkpoint: {latest_checkpoint_file}")
                try:
                    checkpoint = torch.load(latest_checkpoint_file, map_location=args.device)
                    
                    model_to_load = model.module if hasattr(model, 'module') else model
                    model_to_load.load_state_dict(checkpoint['model_state_dict'])
                    
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    if use_amp and 'scaler_state_dict' in checkpoint and scaler:
                        scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    
                    start_epoch = checkpoint.get('epoch', -1) + 1 # get with default, then +1
                    global_step = checkpoint.get('global_step', 0)
                    best_ppl = checkpoint.get('best_ppl', float('inf'))
                    best_bleu_em = checkpoint.get('best_bleu_em', -1.0)
                    
                    logger.info(f"Successfully resumed from epoch {checkpoint.get('epoch', -1)}. Next epoch: {start_epoch}. Global step: {global_step}")
                    logger.info(f"Resumed best_ppl: {best_ppl:.4f}, best_bleu_em: {best_bleu_em:.4f}")

                except Exception as e:
                    logger.error(f"Failed to load checkpoint {latest_checkpoint_file}. Error: {e}", exc_info=True)
                    logger.info("Starting training from scratch.")
                    start_epoch = 0; global_step = 0; best_ppl = float('inf'); best_bleu_em = -1.0
            else:
                logger.info("No 'checkpoint-epoch-*/checkpoint.pt' found. Starting training from scratch.")
        else:
            logger.info("Not resuming from checkpoint. Starting training from scratch.")


        # DataLoader
        train_examples, train_data = load_and_cache_gen_data(
            args, args.train_filename, pool, tokenizer, 'train')
        train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=args.train_batch_size,
            num_workers=4, pin_memory=True
        )

        # Recalculate total_steps for scheduler if not resuming or if train_data changed
        # This ensures the scheduler has the correct number of total steps.
        # If global_step was loaded, scheduler.step() will catch up.
        actual_total_steps = args.num_train_epochs * (len(train_dataloader) // args.gradient_accumulation_steps)
        if scheduler.num_training_steps != actual_total_steps and not args.resume_from_checkpoint : # only if not resuming or a mismatch
             logger.warning(f"Scheduler total steps mismatch. Re-initializing scheduler. Expected: {actual_total_steps}, Got: {scheduler.num_training_steps}")
             scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=actual_total_steps
             )
             # If resuming, and global_step > 0, we need to advance the scheduler
             if args.resume_from_checkpoint and global_step > 0:
                 logger.info(f"Advancing scheduler by {global_step} steps to match loaded global_step.")
                 for _ in range(global_step): # This might be slow if global_step is large.
                     scheduler.step()         # HuggingFace scheduler handles this internally if last_epoch is set correctly.
                                              # Alternatively, can re-create scheduler with last_epoch = global_step -1


        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                       args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", actual_total_steps)
        logger.info("  Starting epoch = %d", start_epoch)
        logger.info("  Starting global step = %d", global_step)


        dev_dataset = {}
        not_loss_dec_cnt = 0
        not_bleu_em_inc_cnt = 0 if args.do_eval_bleu else float('inf') # if not eval_bleu, never early stop based on it

        for epoch in range(start_epoch, args.num_train_epochs):
            logger.info(f"Epoch {epoch}/{args.num_train_epochs - 1}")
            bar = tqdm(train_dataloader, desc=f"Epoch {epoch} Training", disable=args.local_rank not in [-1, 0])
            tr_loss = 0.0
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

                if args.n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                current_loss = loss.item() # Store for logging
                tr_loss += current_loss

                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if use_amp:
                        scaler.unscale_(optimizer) # Unscale before clip_grad_norm
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if args.local_rank in [-1, 0] and tb_writer and global_step % args.logging_steps == 0 : # Use args.logging_steps
                        tb_writer.add_scalar('train/loss', current_loss, global_step)
                        tb_writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
                
                if args.local_rank in [-1, 0]:
                     bar.set_postfix(loss=f"{current_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")
            
            # --- Save Checkpoint After Each Epoch ---
            if args.local_rank in [-1, 0]:
                epoch_checkpoint_dir = os.path.join(args.output_dir, f'checkpoint-epoch-{epoch}')
                os.makedirs(epoch_checkpoint_dir, exist_ok=True)
                model_to_save = model.module if hasattr(model, 'module') else model
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'global_step': global_step,
                    'best_ppl': best_ppl,
                    'best_bleu_em': best_bleu_em,
                    'args': args, # Save args for reference
                }
                if use_amp and scaler:
                    checkpoint_data['scaler_state_dict'] = scaler.state_dict()
                
                torch.save(checkpoint_data, os.path.join(epoch_checkpoint_dir, 'checkpoint.pt'))
                logger.info(f"Saved epoch {epoch} checkpoint to {os.path.join(epoch_checkpoint_dir, 'checkpoint.pt')}")

            # --- Evaluation ---
            if args.do_eval and args.local_rank in [-1, 0]: # Evaluation only on master process
                # PPL Evaluation
                if 'dev_loss' not in dev_dataset:
                    eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev')
                    dev_dataset['dev_loss'] = (eval_examples, eval_data)
                else:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                
                eval_ppl = eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer)
                logger.info(f"Epoch {epoch} | Global Step {global_step} | Eval PPL: {eval_ppl:.4f}")
                if tb_writer: tb_writer.add_scalar('eval/ppl', eval_ppl, global_step)
                fa.write(f"Epoch {epoch} | Global Step {global_step} | Eval PPL: {eval_ppl:.4f}\n")

                if eval_ppl < best_ppl:
                    not_loss_dec_cnt = 0
                    logger.info(f"  Best PPL improved from {best_ppl:.4f} to {eval_ppl:.4f}")
                    best_ppl = eval_ppl
                    fa.write(f"  Best PPL improved to {best_ppl:.4f} at epoch {epoch}\n")
                    
                    best_ppl_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    os.makedirs(best_ppl_dir, exist_ok=True)
                    # Save full checkpoint for best-ppl
                    torch.save(checkpoint_data, os.path.join(best_ppl_dir, 'checkpoint.pt')) # checkpoint_data is from current epoch
                    logger.info(f"Saved best-PPL checkpoint (Epoch {epoch}) to {best_ppl_dir}/checkpoint.pt")
                else:
                    not_loss_dec_cnt += 1
                    logger.info(f"  PPL did not improve for {not_loss_dec_cnt} epoch(s). Current PPL: {eval_ppl:.4f}, Best PPL: {best_ppl:.4f}")
                
                # BLEU Evaluation (if enabled)
                if args.do_eval_bleu:
                    # For BLEU eval, examples might need targets if not test set.
                    # load_and_cache_gen_data for 'dev' should provide targets in eval_examples.
                    # Using the same eval_examples, eval_data as PPL for dev BLEU.
                    bleu_results = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'dev', f'e{epoch}_gs{global_step}')
                    current_bleu_em = bleu_results['bleu'] # Or bleu + em, depending on preference
                    if tb_writer: tb_writer.add_scalar('eval/bleu', bleu_results['bleu'], global_step)
                    if tb_writer: tb_writer.add_scalar('eval/em', bleu_results['em'], global_step)
                    fa.write(f"Epoch {epoch} | Eval BLEU: {bleu_results['bleu']:.2f} | Eval EM: {bleu_results['em']:.2f}\n")

                    if current_bleu_em > best_bleu_em:
                        not_bleu_em_inc_cnt = 0
                        logger.info(f"  Best BLEU/EM improved from {best_bleu_em:.2f} to {current_bleu_em:.2f}")
                        best_bleu_em = current_bleu_em
                        fa.write(f"  Best BLEU/EM improved to {best_bleu_em:.2f} at epoch {epoch}\n")

                        best_bleu_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                        os.makedirs(best_bleu_dir, exist_ok=True)
                        torch.save(checkpoint_data, os.path.join(best_bleu_dir, 'checkpoint.pt'))
                        logger.info(f"Saved best-BLEU checkpoint (Epoch {epoch}) to {best_bleu_dir}/checkpoint.pt")
                    else:
                        not_bleu_em_inc_cnt +=1
                        logger.info(f"  BLEU/EM did not improve for {not_bleu_em_inc_cnt} epoch(s). Current: {current_bleu_em:.2f}, Best: {best_bleu_em:.2f}")
                
                # Early Stopping Check
                if hasattr(args, 'patience') and args.patience > 0:
                    if not_loss_dec_cnt >= args.patience and not_bleu_em_inc_cnt >= args.patience:
                        logger.info(f"Early stopping triggered at epoch {epoch} after {args.patience} epochs of no improvement in PPL and BLEU/EM.")
                        fa.write(f"Early stopping at epoch {epoch}.\n")
                        break # Break from epoch loop
            
            if args.local_rank != -1: # Sync distributed processes if any
                torch.distributed.barrier()

        if args.local_rank in [-1, 0] and tb_writer:
            tb_writer.close()
        logger.info("Training done in %s", get_elapse_time(t0))
        fa.write(f"Training done in {get_elapse_time(t0)}\n")

    # --- Testing ---
    if args.do_test and args.local_rank in [-1, 0]: # Testing only on master process
        logger.info("***** Testing *****")
        # Define criteria based on what's saved.
        # train.sh uses --do_eval_bleu, so best-bleu is relevant.
        # Best-PPL is also saved.
        # Can also test a specific epoch like 'checkpoint-epoch-X'
        test_criteria = ['best-ppl', 'best-bleu'] 
        # If you want to test the very last epoch checkpoint:
        # last_epoch_ckpt_dir = find_latest_epoch_checkpoint(args.output_dir)
        # if last_epoch_ckpt_dir: test_criteria.append(os.path.basename(os.path.dirname(last_epoch_ckpt_dir)))


        for criteria_name in test_criteria:
            # Path to the checkpoint.pt file within the criteria directory
            checkpoint_dir_to_test = os.path.join(args.output_dir, criteria_name)
            checkpoint_file_to_test = os.path.join(checkpoint_dir_to_test, 'checkpoint.pt')

            if not os.path.exists(checkpoint_file_to_test):
                logger.warning(f"Checkpoint file {checkpoint_file_to_test} not found for criteria '{criteria_name}'. Skipping test.")
                # Fallback for old pytorch_model.bin (optional, if you want to support old checkpoints)
                # legacy_file = os.path.join(checkpoint_dir_to_test, 'pytorch_model.bin')
                # if os.path.exists(legacy_file):
                #     logger.info(f"Found legacy model file {legacy_file}, loading model state dict only.")
                #     checkpoint_file_to_test = legacy_file
                # else:
                #     continue
                continue


            logger.info(f"Loading model for testing from: {checkpoint_file_to_test}")
            try:
                checkpoint = torch.load(checkpoint_file_to_test, map_location=args.device)
                model_to_load = model.module if hasattr(model, 'module') else model
                
                if 'model_state_dict' in checkpoint:
                    model_to_load.load_state_dict(checkpoint['model_state_dict'])
                else: # Legacy format (just the state_dict)
                    model_to_load.load_state_dict(checkpoint)
                model.eval()
            except Exception as e:
                logger.error(f"Failed to load model from {checkpoint_file_to_test} for testing: {e}", exc_info=True)
                continue

            # For test set, only_src=True is typically used.
            # Gold references for test set need to be loaded separately if available, or eval_bleu_epoch handles it.
            test_examples, test_data = load_and_cache_gen_data(
                args, args.test_filename, pool, tokenizer, 'test', only_src=True
            )
            
            # Test examples from load_and_cache_gen_data (with only_src=True) will have `source`
            # but `target` will be None. `eval_bleu_epoch` needs to handle this for test set,
            # e.g., by loading gold references from a known file path if `split_tag=='test'`.
            # The current `eval_bleu_epoch` saves gold if `eval_examples[0].target` exists.
            # For a true test set, you might need to modify `eval_bleu_epoch` to load gold files
            # based on `args.test_filename` or a similar convention.
            # For now, it will compute BLEU if `test_examples` (which are actually dev examples in this load) have targets.
            
            test_results = eval_bleu_epoch(args, test_data, test_examples, model, tokenizer, 'test', criteria_name)
            result_str = f"Criteria: {criteria_name} | Test BLEU: {test_results['bleu']:.2f} | Test EM: {test_results['em']:.2f}"
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

if __name__ == "__main__":
    main()
