#!/usr/bin/env python3
"""
FSDP Gradient Ascent Unlearning for GPT-2 (NTP, loss only on generation tokens)

Dataset:
- Pickle file containing either:
  (A) list[dict] where each dict has keys: {"prompt","generation","label"}
  (B) dict[str, dict] with same inner keys (we'll use values()).

Unlearning:
- forget set: label == 1  -> gradient ASCENT (maximize loss on generation tokens)
- retain set: label == 0  -> (optional) normal gradient DESCENT to preserve utility
  By default retain_weight=0 => simplest "GA only on forget set".

Checkpoint:
- Pass --model_name_or_path as either:
  1) a HF model name (e.g. "gpt2"), OR
  2) a directory containing a saved state dict file like "pytorch_model.bin" or "pytorch_model.pt"
     (as in your screenshot). We'll instantiate from --base_model (default: "gpt2") and load weights.

FSDP:
- Run with torchrun --nproc_per_node=2
"""

from __future__ import annotations
import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
import argparse
import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial




# ----------------------------
# Utilities
# ----------------------------

def is_rank0() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def setup_distributed() -> Tuple[int, int, int]:
    """Returns (rank, world_size, local_rank)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        # Single process fallback
        return 0, 1, 0


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_model_state_file(ckpt_dir: Union[str, Path]) -> Optional[Path]:
    """Find a pytorch_model.* file inside ckpt_dir."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return None
    # common names from HF / custom saves
    candidates = []
    for pat in ["pytorch_model.bin", "pytorch_model.pt", "model.safetensors"]:
        p = ckpt_dir / pat
        if p.exists():
            return p
    # fallback: anything starting with pytorch_model
    for p in ckpt_dir.iterdir():
        if p.is_file() and p.name.startswith("pytorch_model"):
            candidates.append(p)
    return sorted(candidates)[0] if candidates else None


def maybe_load_meta(ckpt_dir: Union[str, Path]) -> Dict[str, Any]:
    p = Path(ckpt_dir) / "meta.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def safe_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


# ----------------------------
# Dataset
# ----------------------------

@dataclass
class Example:
    prompt: str
    generation: str
    label: int  # 1=forget, 0=retain


class PicklePromptGenDataset(Dataset):
    def __init__(self, data_path: str, target_label: int):
        with open(data_path, "rb") as f:
            obj = pickle.load(f)

        if isinstance(obj, dict):
            items = list(obj.values())
        elif isinstance(obj, list):
            items = obj
        else:
            raise ValueError(f"Unsupported pickle type: {type(obj)} (expect list or dict)")

        filtered: List[Example] = []
        for d in items:
            if not isinstance(d, dict):
                continue
            if "prompt" not in d or "generation" not in d or "label" not in d:
                continue
            if int(d["label"]) != int(target_label):
                continue
            filtered.append(Example(prompt=str(d["prompt"]), generation=str(d["generation"]), label=int(d["label"])))

        if len(filtered) == 0:
            raise ValueError(f"No examples found with label={target_label} in {data_path}")

        self.data = filtered

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Example:
        return self.data[idx]


def build_prompt_gen_features(
    ex: Example,
    tokenizer,
    seq_len: int,
    add_eos: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize prompt and generation separately, then concat.
    Labels: -100 for prompt tokens, generation token ids for generation tokens.
    Truncation rule:
      - prioritize keeping generation tokens; truncate prompt first from the left.
    """
    prompt_ids = tokenizer.encode(ex.prompt, add_special_tokens=False)
    gen_text = ex.generation + (tokenizer.eos_token if (add_eos and tokenizer.eos_token is not None) else "")
    gen_ids = tokenizer.encode(gen_text, add_special_tokens=False)

    # If generation alone too long, keep last seq_len tokens of generation
    if len(gen_ids) > seq_len:
        gen_ids = gen_ids[-seq_len:]
        prompt_ids = []

    # Otherwise truncate prompt to fit
    max_prompt = max(0, seq_len - len(gen_ids))
    if len(prompt_ids) > max_prompt:
        prompt_ids = prompt_ids[-max_prompt:]  # truncate from left

    input_ids = prompt_ids + gen_ids
    labels = ([-100] * len(prompt_ids)) + gen_ids

    # pad
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # GPT-2 often has no pad token; set to eos for padding usage
        pad_id = tokenizer.eos_token_id

    attn = [1] * len(input_ids)
    if len(input_ids) < seq_len:
        pad_n = seq_len - len(input_ids)
        input_ids = input_ids + [pad_id] * pad_n
        labels = labels + [-100] * pad_n
        attn = attn + [0] * pad_n

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class Collator:
    def __init__(self, tokenizer, seq_len: int):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __call__(self, batch: List[Example]) -> Dict[str, torch.Tensor]:
        feats = [build_prompt_gen_features(ex, self.tokenizer, self.seq_len) for ex in batch]
        return {
            "input_ids": torch.stack([f["input_ids"] for f in feats], dim=0),
            "attention_mask": torch.stack([f["attention_mask"] for f in feats], dim=0),
            "labels": torch.stack([f["labels"] for f in feats], dim=0),
        }


# ----------------------------
# Model / FSDP
# ----------------------------

def load_model_and_tokenizer(
    model_name_or_path: str,
    base_model: str,
    torch_dtype: torch.dtype,
) -> Tuple[torch.nn.Module, Any, Any]:
    """
    If model_name_or_path is a directory with pytorch_model.*:
      - instantiate from base_model and load state dict
    Else:
      - load from_pretrained(model_name_or_path)
    """
    p = Path(model_name_or_path)
    state_file = find_model_state_file(p) if p.exists() else None

    if state_file is not None:
        # local checkpoint dir
        config = AutoConfig.from_pretrained(base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            config=config,
            torch_dtype=torch_dtype,
        )

        # load state dict
        sd = torch.load(state_file, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if is_rank0():
            print(f"[load] Loaded state dict from {state_file}")
            if missing:
                print(f"[load] missing keys (showing up to 20): {missing[:20]}")
            if unexpected:
                print(f"[load] unexpected keys (showing up to 20): {unexpected[:20]}")
        return model, tokenizer, config

    # HF path/name
    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
    )
    return model, tokenizer, config


def wrap_fsdp(model: torch.nn.Module, bf16: bool = True) -> torch.nn.Module:
    """
    FSDP wrap GPT2 blocks.
    Compatible with torch versions where transformer_auto_wrap_policy is a policy fn.
    """
    # GPT-2 block class
    block_cls = set()
    try:
        from transformers.models.gpt2.modeling_gpt2 import GPT2Block
        block_cls = {GPT2Block}
    except Exception:
        block_cls = set()

    mp = MixedPrecision(
        param_dtype=torch.bfloat16 if bf16 else torch.float16,
        reduce_dtype=torch.bfloat16 if bf16 else torch.float16,
        buffer_dtype=torch.bfloat16 if bf16 else torch.float16,
    )

    # IMPORTANT: make a policy via partial (newer torch API shape)
    auto_wrap_policy = None
    if block_cls:
        auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=block_cls)

    # If we couldn't import GPT2Block, wrap whole model
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,  # None => whole-model wrap
        mixed_precision=mp,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
    )

# ----------------------------
# Checkpoint save/load
# ----------------------------

def save_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    out_dir: str,
    step: int,
    epoch: int,
    extra_meta: Optional[Dict[str, Any]] = None,
):
    out = Path(out_dir) / f"step_{step:08d}"
    out.mkdir(parents=True, exist_ok=True)

    # full (gathered) state dict on rank0
    full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
        state = model.state_dict()

    if is_rank0():
        torch.save(state, out / "pytorch_model.bin")
        torch.save(optimizer.state_dict(), out / "optimizer.pt")
        if scheduler is not None:
            torch.save(scheduler.state_dict(), out / "scheduler.pt")

        meta = {
            "global_step": step,
            "epoch": epoch,
        }
        if extra_meta:
            meta.update(extra_meta)
        (out / "meta.json").write_text(json.dumps(meta, indent=2))

    safe_barrier()


def try_resume_optimizer_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    ckpt_dir: str,
):
    ckpt = Path(ckpt_dir)
    opt_p = ckpt / "optimizer.pt"
    sch_p = ckpt / "scheduler.pt"

    if opt_p.exists():
        try:
            optimizer.load_state_dict(torch.load(opt_p, map_location="cpu"))
            if is_rank0():
                print(f"[resume] Loaded optimizer from {opt_p}")
        except Exception as e:
            if is_rank0():
                print(f"[resume] Failed to load optimizer: {e}")

    if scheduler is not None and sch_p.exists():
        try:
            scheduler.load_state_dict(torch.load(sch_p, map_location="cpu"))
            if is_rank0():
                print(f"[resume] Loaded scheduler from {sch_p}")
        except Exception as e:
            if is_rank0():
                print(f"[resume] Failed to load scheduler: {e}")


# ----------------------------
# Main train
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True, help="Pickle with list/dict of {prompt,generation,label}")
    ap.add_argument("--model_name_or_path", type=str, required=True, help="HF name OR checkpoint directory")
    ap.add_argument("--base_model", type=str, default="gpt2", help="Base model if loading from raw state dict dir")
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=2, help="Per-GPU batch size")
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)

    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    # Unlearning weights
    ap.add_argument("--forget_weight", type=float, default=1.0, help="Weight for GA on forget loss")
    ap.add_argument("--retain_weight", type=float, default=0.0, help="Weight for GD on retain loss (0 = GA only)")

    # Scheduler
    ap.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "linear", "none"])
    ap.add_argument("--warmup_steps", type=int, default=100)

    # Resume optimizer/scheduler from checkpoint dir (if present)
    ap.add_argument("--resume_optimizer", action="store_true")

    # Logging/saving
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=20)

    # W&B
    ap.add_argument("--wandb_project", type=str, default="gpt2-unlearning-ga")
    ap.add_argument("--run_name", type=str, default=None)

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rank, world, local_rank = setup_distributed()
    seed_all(args.seed + rank)

    os.makedirs(args.output_dir, exist_ok=True)

    # W&B init (rank0 only)
    wandb = None
    if is_rank0():
        try:
            import wandb as _wandb
            wandb = _wandb
            wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config=vars(args),
            )
        except Exception as e:
            print(f"[wandb] disabled (import/init failed): {e}")
            wandb = None

    # dtype: bf16 on A5000
    torch_dtype = torch.bfloat16

    model, tokenizer, _ = load_model_and_tokenizer(
        args.model_name_or_path,
        base_model=args.base_model,
        torch_dtype=torch_dtype,
    )
    model.cuda()
    model.train()

    model = wrap_fsdp(model, bf16=True)

    # Build datasets
    forget_ds = PicklePromptGenDataset(args.data_path, target_label=1)
    retain_ds = PicklePromptGenDataset(args.data_path, target_label=0) if args.retain_weight > 0 else None

    collate = Collator(tokenizer, args.seq_len)

    forget_sampler = DistributedSampler(forget_ds, num_replicas=world, rank=rank, shuffle=True)
    forget_loader = DataLoader(
        forget_ds,
        batch_size=args.batch_size,
        sampler=forget_sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate,
        drop_last=True,
    )

    if retain_ds is not None:
        retain_sampler = DistributedSampler(retain_ds, num_replicas=world, rank=rank, shuffle=True)
        retain_loader = DataLoader(
            retain_ds,
            batch_size=args.batch_size,
            sampler=retain_sampler,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate,
            drop_last=True,
        )
    else:
        retain_loader = None
        retain_sampler = None

    # Optimizer / scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # total steps (rough estimate; for cosine schedule)
    steps_per_epoch = math.ceil(len(forget_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs

    if args.scheduler == "none":
        scheduler = None
    elif args.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    # If the input is a checkpoint dir and user wants to resume optimizer/scheduler
    ckpt_dir = Path(args.model_name_or_path)
    if args.resume_optimizer and ckpt_dir.is_dir():
        try_resume_optimizer_scheduler(optimizer, scheduler, str(ckpt_dir))

    # Try to read prior meta for initial global_step/epoch (nice-to-have)
    meta = maybe_load_meta(args.model_name_or_path) if ckpt_dir.is_dir() else {}
    global_step = int(meta.get("global_step", 0))
    start_epoch = int(meta.get("epoch", 0))

    if is_rank0():
        print(f"[data] forget: {len(forget_ds)} examples")
        if retain_ds is not None:
            print(f"[data] retain: {len(retain_ds)} examples")
        print(f"[train] start_epoch={start_epoch} global_step={global_step} total_steps~={total_steps}")

    # Training loop
    model.train()
    accum = 0
    optimizer.zero_grad(set_to_none=True)

    # If retain is enabled, we cycle it to match forget iterations
    retain_iter = iter(retain_loader) if retain_loader is not None else None

    for epoch in range(start_epoch, args.epochs):
        forget_sampler.set_epoch(epoch)
        if retain_sampler is not None:
            retain_sampler.set_epoch(epoch)

        pbar = tqdm(forget_loader, disable=not is_rank0(), desc=f"epoch {epoch}")
        for it, batch_forget in enumerate(pbar):
            # Move to GPU
            batch_forget = {k: v.cuda(non_blocking=True) for k, v in batch_forget.items()}

            # Forget loss (generation-token-only labels already set)
            out_f = model(**batch_forget)
            forget_loss = out_f.loss

            # Gradient ascent => maximize forget_loss => minimize (-forget_loss)
            total_loss = -args.forget_weight * forget_loss

            retain_loss = None
            if retain_iter is not None:
                try:
                    batch_retain = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    batch_retain = next(retain_iter)

                batch_retain = {k: v.cuda(non_blocking=True) for k, v in batch_retain.items()}
                out_r = model(**batch_retain)
                retain_loss = out_r.loss
                total_loss = total_loss + (args.retain_weight * retain_loss)

            # Normalize for grad accumulation
            total_loss = total_loss / args.grad_accum
            total_loss.backward()
            accum += 1

            if accum >= args.grad_accum:
                # optional grad clipping (need to use FSDP's clip util? plain works OK here)
                if args.max_grad_norm > 0:
                    try:
                        model.clip_grad_norm_(args.max_grad_norm)
                    except Exception:
                        clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                global_step += 1

                # Logging
                if global_step % args.log_every == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    log = {
                        "step": global_step,
                        "lr": lr,
                        "forget_loss": float(forget_loss.detach().cpu()),
                    }
                    if retain_loss is not None:
                        log["retain_loss"] = float(retain_loss.detach().cpu())

                    if is_rank0():
                        pbar.set_postfix({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in log.items() if k != "step"})
                        if wandb is not None:
                            wandb.log(log, step=global_step)

                # Save
                if global_step % args.save_every == 0:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        out_dir=args.output_dir,
                        step=global_step,
                        epoch=epoch,
                        extra_meta={"base_model": args.base_model},
                    )

        # end epoch save
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            out_dir=args.output_dir,
            step=global_step,
            epoch=epoch + 1,
            extra_meta={"base_model": args.base_model},
        )

    if is_rank0() and wandb is not None:
        wandb.finish()

    safe_barrier()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
