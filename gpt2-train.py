#!/usr/bin/env python3
"""
Train GPT-2 on next-token prediction (causal LM) from a pickle dict dataset.

Your pickle file should be a dict like:
  data_dict[k] = {
      "generation": <str>,
      "label": <ignored>
  }

We use "generation" text only.

Features:
- FSDP sharding across GPUs (torchrun)
- LoRA via PEFT (default: freeze base, train adapters only)
- Mixed precision (bf16 on A5000)
- tqdm progress bar (rank0 only)
- wandb logging (rank0 only)
- checkpoint saving with full state dict gathered on rank0

Run:
  torchrun --nproc_per_node=2 train_gpt2_fsdp_lora.py \
    --data_path /path/to/data.pkl \
    --output_dir ./ckpts_gpt2 \
    --wandb_project gpt2-next-token \
    --run_name a5000-fsdp-lora \
    --epochs 1 --batch_size 2 --grad_accum 8 --seq_len 512 --lr 2e-4
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import math
import time
import pickle
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from tqdm import tqdm

# HF / PEFT
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model

# FSDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import functools
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
# Activation checkpointing (optional)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

try:
    import wandb
except Exception:
    wandb = None

IGNORE_INDEX = -100
# -----------------------------
# utils
# -----------------------------
def is_dist():
    return dist.is_available() and dist.is_initialized()

def rank():
    return dist.get_rank() if is_dist() else 0

def world_size():
    return dist.get_world_size() if is_dist() else 1

def is_rank0():
    return rank() == 0

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ddp_barrier():
    if is_dist():
        dist.barrier()


def init_distributed():
    """
    torchrun sets: RANK, LOCAL_RANK, WORLD_SIZE
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        # single-process fallback
        pass


# -----------------------------
# dataset
# -----------------------------

# class GPT2BlockDataset(Dataset):
#     def __init__(
#         self,
#         data_path: str,
#         tokenizer,
#         seq_len: int = 512,
#         max_samples: Optional[int] = None,
#         shuffle_texts: bool = True,
#         add_eos: bool = True,
#         tok_batch_size: int = 256,     # ✅ controls speed/memory of tokenization
#         show_tokenize_pbar: bool = True,
#     ):
#         super().__init__()
#         self.seq_len = seq_len
#         self.tokenizer = tokenizer
#         self.add_eos = add_eos
        
#         with open(data_path, "rb") as f:
#             data_dict = pickle.load(f)

#         texts = []
#         for _, v in tqdm(data_dict.items(), desc="Get Data"):
#             if isinstance(v, dict):
#                 t = v.get("generation", None)
#                 if isinstance(t, str) and t.strip():
#                     texts.append(t)

#         if shuffle_texts:
#             random.shuffle(texts)
#         if max_samples is not None:
#             texts = texts[:max_samples]

#         eos = tokenizer.eos_token_id

#         all_ids: List[int] = []
#         it = range(0, len(texts), tok_batch_size)
#         print('start tokenizing')

#         pbar = tqdm(
#             it,
#             desc="Tokenizing",
#             dynamic_ncols=True,
#             disable=(not show_tokenize_pbar) or (not is_rank0()),
#         )

#         for i in pbar:
#             batch_texts = texts[i : i + tok_batch_size]
#             enc = tokenizer(batch_texts, add_special_tokens=False)
#             # enc["input_ids"] is a list[list[int]]
#             for ids in enc["input_ids"]:
#                 if add_eos and eos is not None:
#                     ids = ids + [eos]
#                 all_ids.extend(ids)

#         # chunk into fixed blocks
#         n_blocks = len(all_ids) // seq_len
#         self.blocks = [
#             torch.tensor(all_ids[i * seq_len : (i + 1) * seq_len], dtype=torch.long)
#             for i in range(n_blocks)
#         ]

#         if is_rank0():
#             print(f"[dataset] texts={len(texts)} total_tokens={len(all_ids)} blocks={len(self.blocks)} seq_len={seq_len}")

#     def __len__(self):
#         return len(self.blocks)

#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         x = self.blocks[idx]
#         return {"input_ids": x, "labels": x.clone()}

def pad_collate(batch, pad_id, label_pad_id=IGNORE_INDEX):
    max_len = max(x["input_ids"].size(0) for x in batch)

    def pad_1d(t, pad_value):
        return torch.nn.functional.pad(t, (0, max_len - t.size(0)), value=pad_value)

    input_ids = torch.stack([pad_1d(x["input_ids"], pad_id) for x in batch])
    attention_mask = torch.stack([pad_1d(x["attention_mask"], 0) for x in batch])
    labels = torch.stack([pad_1d(x["labels"], label_pad_id) for x in batch])

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}




class PromptGenCausalDataset(Dataset):
    """
    Each item supervises ONLY the generation tokens.
    """
    def __init__(self, data_list, tokenizer, max_length=1024, separator="\n"):
        self.data = data_list
        self.tok = tokenizer
        self.max_length = max_length
        self.sep = separator

        # for GPT2-like tokenizers
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        prompt = ex["prompt"]
        gen = ex["generation"]

        # Tokenize separately so we can mask prompt labels cleanly
        prompt_ids = self.tok(prompt + self.sep, add_special_tokens=False)["input_ids"]
        gen_ids = self.tok(gen, add_special_tokens=False)["input_ids"]

        # Optionally ensure an EOS at the end of the generation
        if self.tok.eos_token_id is not None:
            gen_ids = gen_ids + [self.tok.eos_token_id]

        input_ids = prompt_ids + gen_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + gen_ids[:]  # supervise only generation

        # Truncate to max_length.
        # IMPORTANT: usually you want to KEEP the generation, so truncate from the LEFT.
        if len(input_ids) > self.max_length:
            overflow = len(input_ids) - self.max_length
            input_ids = input_ids[overflow:]
            labels = labels[overflow:]

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@dataclass
class Collate:
    """
    GPT-2 doesn't need padding if every example is fixed seq_len.
    """
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
        labels = torch.stack([b["labels"] for b in batch], dim=0)
        return {"input_ids": input_ids, "labels": labels}


# -----------------------------
# model building
# -----------------------------
def build_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    # GPT-2 has no pad token by default; set to eos to avoid warnings if needed.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = GPT2LMHeadModel.from_pretrained(args.model_name)

    if args.use_lora:
        # Typical GPT-2 attention module uses "c_attn" and "c_proj"
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["c_attn", "c_proj"],
        )
        model = get_peft_model(base, lora_cfg)

        if args.freeze_base:
            # PEFT already makes only LoRA trainable; this is just extra safety.
            for n, p in model.named_parameters():
                if "lora_" not in n:
                    p.requires_grad = False
    else:
        model = base

    return tokenizer, model


def maybe_apply_activation_ckpt(model: nn.Module, args):
    if not args.activation_checkpointing:
        return model

    # Wrap transformer blocks with activation checkpointing
    # For HF GPT-2, blocks are model.transformer.h (GPT2Block)
    try:
        from transformers.models.gpt2.modeling_gpt2 import GPT2Block
        check_fn = lambda m: isinstance(m, GPT2Block)
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(
                m, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            ),
            check_fn=check_fn,
        )
        if is_rank0():
            print("[ckpt] activation checkpointing enabled for GPT2Block")
    except Exception as e:
        if is_rank0():
            print(f"[ckpt] activation checkpointing requested but failed: {e}")
    return model


def wrap_fsdp(model, args):
    use_bf16 = args.precision == "bf16"
    mp = MixedPrecision(
        param_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        reduce_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        buffer_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={GPT2Block},
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
    )
    return model


# -----------------------------
# metrics
# -----------------------------
@torch.no_grad()
def token_accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    logits: [B, T, V]
    labels: [B, T]
    GPT-2 loss uses shift: predict token t+1 from position t.
    So compare logits[:, :-1] with labels[:, 1:].
    """
    preds = torch.argmax(logits[:, :-1, :], dim=-1)      # [B, T-1]
    gold = labels[:, 1:]                                 # [B, T-1]
    correct = (preds == gold).float().sum()
    total = torch.numel(gold)
    return correct / max(total, 1)


def grad_norm(parameters, norm_type=2.0):
    params = [p for p in parameters if p.grad is not None]
    if len(params) == 0:
        return torch.tensor(0.0, device="cuda")
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type) for p in params]),
        norm_type,
    )
    return total_norm


# -----------------------------
# checkpointing
# -----------------------------
def save_checkpoint(output_dir: str, model: nn.Module, optimizer, scheduler, step: int, epoch: int):
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, f"step_{step:08d}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Gather full state dict on rank0 only
    full_state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_cfg):
        state = model.state_dict()

    if is_rank0():
        torch.save(state, os.path.join(ckpt_dir, "pytorch_model.bin"))
        torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))
        with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
            json.dump({"step": step, "epoch": epoch}, f, indent=2)


# -----------------------------
# training
# -----------------------------
def train(args):
    init_distributed()
    set_seed(args.seed + rank())

    # wandb on rank0
    if args.use_wandb:
        if wandb is None:
            raise RuntimeError("wandb not installed. `pip install wandb`")
        if is_rank0():
            wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config=vars(args),
            )

    tokenizer, model = build_model_and_tokenizer(args)
    model = maybe_apply_activation_ckpt(model, args)
    model.cuda()

    if is_dist():
        model = wrap_fsdp(model, args)

    # dataset = GPT2BlockDataset(
    #     data_path=args.data_path,
    #     tokenizer=tokenizer,
    #     seq_len=args.seq_len,
    #     max_samples=args.max_samples,
    #     shuffle_texts=True,
    #     add_eos=True,
    # )

    # sampler = DistributedSampler(dataset, shuffle=True) if is_dist() else None
    # loader = DataLoader(
    #     dataset,
    #     batch_size=args.batch_size,
    #     sampler=sampler,
    #     shuffle=(sampler is None),
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     drop_last=True,
    #     collate_fn=Collate(),
    # )

    with open(args.data_path, "rb") as f:
        data_list = pickle.load(f)  # list of {"prompt":..., "generation":...}

    train_ds = PromptGenCausalDataset(data_list, tokenizer, max_length=args.seq_len, separator="\n")
    collate_fn = lambda batch: pad_collate(batch, pad_id=tokenizer.pad_token_id)
    
    sampler = None
    if dist.is_available() and dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds,
            shuffle=True,
            drop_last=False,
        )

    
    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Only optimize trainable params (LoRA adapters if enabled)
    optim_params = [p for p in model.parameters() if p.requires_grad]
    if is_rank0():
        n_trainable = sum(p.numel() for p in optim_params)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"[params] trainable={n_trainable:,} total={n_total:,} ({100.0*n_trainable/n_total:.2f}%)")

    optimizer = torch.optim.AdamW(
        optim_params,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = math.floor(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = None  # using bf16/fp16 autocast; no GradScaler for bf16
    use_bf16 = args.precision == "bf16"
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

    global_step = 0
    model.train()

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        if is_rank0():
            pbar = tqdm(total=len(loader), desc=f"Epoch {epoch+1}/{args.epochs}", dynamic_ncols=True)
        else:
            pbar = None

        running_loss = 0.0
        running_acc = 0.0
        running_tokens = 0
        t0 = time.time()

        optimizer.zero_grad(set_to_none=True)

        for it, batch in enumerate(loader):
            input_ids = batch["input_ids"].cuda(non_blocking=True)
            labels = batch["labels"].cuda(non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                out = model(input_ids=input_ids, labels=labels)
                loss = out.loss / args.grad_accum

            loss.backward()

            # metrics from *unscaled* loss
            with torch.no_grad():
                # token accuracy
                acc = token_accuracy_from_logits(out.logits.detach(), labels)
                # count tokens for throughput (exclude the first token which is not predicted)
                tokens = labels.numel() - labels.size(0)

            running_loss += loss.item() * args.grad_accum
            running_acc += acc.item()
            running_tokens += tokens

            # step if grad_accum boundary
            if (it + 1) % args.grad_accum == 0:
                # grad clip
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(optim_params, args.grad_clip)

                gn = grad_norm(optim_params).detach()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                # reduce metrics across ranks for nicer logging
                loss_t = torch.tensor(running_loss / args.grad_accum, device="cuda")
                acc_t = torch.tensor(running_acc / args.grad_accum, device="cuda")
                tok_t = torch.tensor(float(running_tokens), device="cuda")
                gn_t = gn if isinstance(gn, torch.Tensor) else torch.tensor(float(gn), device="cuda")

                if is_dist():
                    dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
                    dist.all_reduce(acc_t, op=dist.ReduceOp.SUM)
                    dist.all_reduce(tok_t, op=dist.ReduceOp.SUM)
                    dist.all_reduce(gn_t, op=dist.ReduceOp.SUM)
                    loss_t = loss_t / world_size()
                    acc_t = acc_t / world_size()
                    gn_t = gn_t / world_size()

                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                tok_per_s = (tok_t.item() / max(elapsed, 1e-6))

                if is_rank0():
                    if pbar is not None:
                        pbar.update(args.grad_accum)
                        pbar.set_postfix({
                            "step": global_step,
                            "loss": f"{loss_t.item():.4f}",
                            "acc": f"{acc_t.item():.4f}",
                            "lr": f"{lr:.2e}",
                            "gn": f"{gn_t.item():.2f}",
                            "tok/s": f"{tok_per_s:.0f}",
                        })

                    if args.use_wandb:
                        wandb.log({
                            "train/loss": loss_t.item(),
                            "train/token_acc": acc_t.item(),
                            "train/lr": lr,
                            "train/grad_norm": gn_t.item(),
                            "train/tokens_per_sec": tok_per_s,
                            "train/epoch": epoch,
                            "train/step": global_step,
                        }, step=global_step)

                # reset running window
                running_loss = 0.0
                running_acc = 0.0
                running_tokens = 0
                t0 = time.time()

                # checkpoint
                if args.save_every > 0 and global_step % args.save_every == 0:
                    ddp_barrier()
                    if is_dist():
                        save_checkpoint(args.output_dir, model, optimizer, scheduler, global_step, epoch)
                    else:
                        # non-FSDP
                        if is_rank0():
                            ckpt_dir = os.path.join(args.output_dir, f"step_{global_step:08d}")
                            os.makedirs(ckpt_dir, exist_ok=True)
                            torch.save(model.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))
                            torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))
                            with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
                                json.dump({"step": global_step, "epoch": epoch}, f, indent=2)

        if is_rank0() and pbar is not None:
            pbar.close()

        if args.save_at_epoch_end:
            ddp_barrier()
            ckpt_name = f"epoch_{epoch:04d}_step_{global_step:08d}"
            if is_dist():
                save_checkpoint(args.output_dir, model, optimizer, scheduler, global_step, epoch)
            else:
                if is_rank0():
                    print(f"[ckpt] save_at_epoch_end={args.save_at_epoch_end} global_step={global_step}")
                    ckpt_dir = os.path.join(args.output_dir, ckpt_name)
                    os.makedirs(ckpt_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))
                    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))
                    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
                        json.dump({"step": global_step, "epoch": epoch}, f, indent=2)


    if args.use_wandb and is_rank0():
        wandb.finish()

    ddp_barrier()
    if is_rank0():
        print("Done.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--model_name", type=str, default="gpt2")  # or "gpt2-medium" if it fits
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--max_samples", type=int, default=None)

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)

    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    # LoRA
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--freeze_base", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Activation checkpointing
    p.add_argument("--activation_checkpointing", action="store_true")

    # Logging
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="gpt2-next-token")
    p.add_argument("--run_name", type=str, default="fsdp-lora-run")

    # Saving
    p.add_argument("--save_every", type=int, default=200)  # in optimizer steps; 0 disables
    p.add_argument("--save_at_epoch_end", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
