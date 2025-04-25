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

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from models import build_or_load_gen_model
from evaluator import smooth_bleu
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu
from utils import get_filenames, get_elapse_time, load_and_cache_gen_data
from configs import add_args, set_seed, set_dist

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logger.info(args)
    t0 = time.time()

    # --- AMP initialization ---
    use_amp = args.fp16 and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        logger.info(">>> Mixed Precision Training: FP16 ENABLED")

    # Setup
    set_dist(args)
    set_seed(args)
    config, model, tokenizer = build_or_load_gen_model(args)
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    pool = multiprocessing.Pool(args.cpu_cont)
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(
        args.data_dir, args.task, args.sub_task)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train:
        # TensorBoard
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = f"{args.summary_dir}/{'/'.join(args.output_dir.split('/')[1:])}"
            tb_writer = SummaryWriter(summary_fn)

        # DataLoader
        train_examples, train_data = load_and_cache_gen_data(
            args, args.train_filename, pool, tokenizer, 'train')
        train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler,
            batch_size=args.train_batch_size,
            num_workers=4, pin_memory=True
        )

        # Optimizer & Scheduler
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon
        )
        total_steps = args.num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps
        )

        global_step = 0
        best_ppl = float('inf')
        dev_dataset = {}

        # Training loop
        for epoch in range(int(args.start_epoch), args.num_train_epochs):
            bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
            model.train()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(args.device) for t in batch)
                source_ids, target_ids = batch
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                target_mask = target_ids.ne(tokenizer.pad_token_id)

                # Forward
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(
                            input_ids=source_ids,
                            attention_mask=source_mask,
                            labels=target_ids,
                            decoder_attention_mask=target_mask
                        )
                        loss = outputs.loss
                else:
                    outputs = model(
                        input_ids=source_ids,
                        attention_mask=source_mask,
                        labels=target_ids,
                        decoder_attention_mask=target_mask
                    )
                    loss = outputs.loss

                # Multi-GPU and gradient accumulation
                if args.n_gpu > 1:
                    loss = loss.mean()
                loss = loss / args.gradient_accumulation_steps

                # Backward
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Step
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
                    optimizer.zero_grad()
                    global_step += 1

                bar.set_postfix(loss=loss.item())

            # Evaluation and checkpointing
            if args.do_eval:
                eval_examples, eval_data = dev_dataset.get('dev_loss',
                    load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev')
                )
                dev_dataset['dev_loss'] = (eval_examples, eval_data)
                eval_ppl = eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer)
                logger.info(f"Epoch {epoch} | eval_ppl: {eval_ppl}")
                if eval_ppl < best_ppl:
                    best_ppl = eval_ppl
                    out_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    os.makedirs(out_dir, exist_ok=True)
                    torch.save(
                        (model.module if hasattr(model, 'module') else model).state_dict(),
                        os.path.join(out_dir, 'pytorch_model.bin')
                    )
                    logger.info("Saved best-ppl checkpoint to %s", out_dir)

        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()
        logger.info("Training done in %s", get_elapse_time(t0))

    # Testing
    if args.do_test:
        logger.info("***** Testing *****")
        for criteria in ['best-bleu']:
            ckpt = os.path.join(args.output_dir, f'checkpoint-{criteria}', 'pytorch_model.bin')
            model.load_state_dict(torch.load(ckpt))
            eval_examples, eval_data = load_and_cache_gen_data(
                args, args.test_filename, pool, tokenizer, 'test', only_src=True
            )
            result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'test', criteria)
            logger.info(f"Test {criteria}: {result}")

if __name__ == "__main__":
    main()
