#!/usr/bin/env python3
"""
FSDP NPO-style unlearning for GPT-2.

Data format: list[dict] with keys:
  - prompt: str
  - generation: str
  - label: int (1=forget, 0=retain)

Behavior:
  - Only supervise generation tokens (prompt tokens masked in labels)
  - NPO objective implemented as DPO-style preference loss:
        chosen = retain, rejected = forget
    with a frozen reference model initialized to the same starting checkpoint.

Checkpoint loading:
  - Load base model first (e.g., "gpt2"), then load finetuned weights from:
        <ckpt_dir>/pytorch_model.bin
  - Optionally resume optimizer/scheduler state from:
        <ckpt_dir>/optimizer.pt, <ckpt_dir>/scheduler.pt

Run with torchrun (2 GPUs) + FSDP.
"""

import os
import json
import math
import time
import pickle
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from tqdm.auto import tqdm

import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
from transformers.models.gpt2.modeling_gpt2 import GPT2Block


# ----------------------------
# Utilities
# ----------------------------

def is_main() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def setup_distributed():
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def read_list_of_dicts(path: str) -> List[Dict[str, Any]]:
    """
    Loads list[dict] from:
      - .pt/.pth (torch.load)
      - .pkl/.pickle (pickle)
      - .json (expects a JSON array)
      - .jsonl (one JSON obj per line)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()
    if ext in [".pt", ".pth"]:
        obj = torch.load(path, map_location="cpu")
        if not isinstance(obj, list):
            raise ValueError(f"Expected list in {path}, got {type(obj)}")
        return obj
    if ext in [".pkl", ".pickle"]:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, list):
            raise ValueError(f"Expected list in {path}, got {type(obj)}")
        return obj
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, list):
            raise ValueError(f"Expected JSON array list in {path}, got {type(obj)}")
        return obj
    if ext == ".jsonl":
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out

    raise ValueError(f"Unsupported file type: {ext}")


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Token accuracy over labels != -100
    logits: [B, T, V], labels: [B, T]
    """
    with torch.no_grad():
        pred = torch.argmax(logits, dim=-1)
        mask = labels != -100
        correct = (pred == labels) & mask
        denom = mask.sum().clamp_min(1)
        return correct.sum().float() / denom.float()


def grad_global_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        # grads can be sharded; .data is local shard
        param_norm = p.grad.data.float().norm(2)
        total += param_norm.item() ** 2
    return math.sqrt(total)


# ----------------------------
# Dataset + Collator
# ----------------------------

class PromptGenDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        # minimal validation
        return {
            "prompt": r["prompt"],
            "generation": r["generation"],
            "label": int(r["label"]),
        }


@dataclass
class CollateCfg:
    max_length: int
    add_eos: bool = True


class PromptGenCollator:
    def __init__(self, tokenizer: AutoTokenizer, cfg: CollateCfg):
        self.tok = tokenizer
        self.cfg = cfg

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = [b["prompt"] for b in batch]
        gens = [b["generation"] for b in batch]

        # tokenize prompt alone to know prompt length in tokens
        prompt_enc = self.tok(
            prompts,
            add_special_tokens=False,
            truncation=True,
            max_length=self.cfg.max_length,
            padding=False,
            return_tensors=None,
        )

        # tokenize full prompt+generation
        full_texts = [p + g for p, g in zip(prompts, gens)]
        full_enc = self.tok(
            full_texts,
            add_special_tokens=False,
            truncation=True,
            max_length=self.cfg.max_length - (1 if self.cfg.add_eos else 0),
            padding=False,
            return_tensors=None,
        )

        input_ids_list = []
        labels_list = []
        for i in range(len(batch)):
            inp = full_enc["input_ids"][i]
            prm = prompt_enc["input_ids"][i]

            if self.cfg.add_eos:
                inp = inp + [self.tok.eos_token_id]

            # labels = input_ids, but mask prompt tokens
            labels = inp.copy()
            prm_len = min(len(prm), len(labels))
            for j in range(prm_len):
                labels[j] = -100

            input_ids_list.append(torch.tensor(inp, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        # pad
        input_ids = nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tok.pad_token_id
        )
        labels = nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )
        attention_mask = (input_ids != self.tok.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ----------------------------
# Preference objective (NPO-style)
# ----------------------------

def sequence_logp_from_labels(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Sum log-prob over positions where labels != -100.
    logits: [B, T, V], labels: [B, T]
    returns: [B] logp
    """
    # shift for causal LM: predict token t from logits at t-1
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    log_probs = torch.log_softmax(shift_logits, dim=-1)  # [B, T-1, V]
    mask = shift_labels != -100
    # replace -100 with 0 for gather
    gather_labels = shift_labels.masked_fill(~mask, 0).unsqueeze(-1)  # [B, T-1, 1]
    token_logp = log_probs.gather(-1, gather_labels).squeeze(-1)      # [B, T-1]
    token_logp = token_logp * mask.float()
    return token_logp.sum(dim=-1)  # [B]


@torch.no_grad()
def forward_logp(model: nn.Module, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    returns:
      - seq_logp: [B]
      - token_acc: scalar tensor
    """
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        use_cache=False,
    )
    logits = out.logits
    seq_logp = sequence_logp_from_labels(logits, batch["labels"])
    acc = masked_token_accuracy(logits[:, :-1, :], batch["labels"][:, 1:])
    return seq_logp, acc


def npo_loss(
    pi_logp_chosen: torch.Tensor,
    pi_logp_rejected: torch.Tensor,
    ref_logp_chosen: torch.Tensor,
    ref_logp_rejected: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DPO-style preference loss (retain preferred over forget):
      logits = beta * [ (pi - ref)_chosen - (pi - ref)_rejected ]
      loss = -log(sigmoid(logits))

    Returns:
      loss (scalar), pref_accuracy (scalar)
    """
    logits = beta * ((pi_logp_chosen - ref_logp_chosen) - (pi_logp_rejected - ref_logp_rejected))
    loss = -torch.nn.functional.logsigmoid(logits).mean()
    pref_acc = (logits > 0).float().mean()
    return loss, pref_acc


# ----------------------------
# FSDP wrapping + state saving
# ----------------------------

def wrap_fsdp(model: nn.Module, mp: Optional[MixedPrecision], cpu_offload: bool) -> nn.Module:
    import functools
    auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={GPT2Block})
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
    )


def load_base_then_finetuned(base_model: str, ckpt_dir: str, dtype: torch.dtype) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    ft_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    if not os.path.exists(ft_path):
        raise FileNotFoundError(f"Missing {ft_path}")

    state = torch.load(ft_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if is_main():
        print(f"[load] finetuned weights loaded from {ft_path}")
        if missing:
            print(f"[load] missing keys (first 20): {missing[:20]}")
        if unexpected:
            print(f"[load] unexpected keys (first 20): {unexpected[:20]}")
    return model


def maybe_resume_opt_sched(ckpt_dir: str, optimizer, scheduler, resume_optimizer: bool):
    if not resume_optimizer:
        return
    opt_path = os.path.join(ckpt_dir, "optimizer.pt")
    sch_path = os.path.join(ckpt_dir, "scheduler.pt")
    if os.path.exists(opt_path):
        optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
        if is_main():
            print(f"[resume] loaded optimizer state from {opt_path}")
    if os.path.exists(sch_path) and scheduler is not None:
        scheduler.load_state_dict(torch.load(sch_path, map_location="cpu"))
        if is_main():
            print(f"[resume] loaded scheduler state from {sch_path}")


def save_full_model_fsdp(model: FSDP, out_dir: str, tag: str):
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"pytorch_model_{tag}.bin")

    # full state dict gathering
    full_state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_cfg):
        state = model.state_dict()

    if is_main():
        torch.save(state, save_path)
        print(f"[save] full model state saved to {save_path}")


# ----------------------------
# Main
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True, help="Path to list[dict] data file (.pt/.pkl/.json/.jsonl)")
    p.add_argument("--ckpt_dir", type=str, required=True, help="Directory containing pytorch_model.bin (and optionally optimizer.pt/scheduler.pt)")
    p.add_argument("--base_model", type=str, default="gpt2", help="Base model name (default: gpt2)")
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=2, help="Per-GPU batch size (retain and forget each use this)")
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=-1, help="If >0, stop after this many optimizer steps")

    p.add_argument("--beta", type=float, default=0.1, help="Preference strength for NPO/DPO-style loss")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--cpu_offload", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--wandb_project", type=str, default="npo-unlearning")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)

    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--resume_optimizer", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    setup_distributed()
    set_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))

    # dtype
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # load data
    rows = read_list_of_dicts(args.data_path)
    forget_rows = [r for r in rows if int(r["label"]) == 1]
    retain_rows = [r for r in rows if int(r["label"]) == 0]
    if is_main():
        print(f"[data] total={len(rows)} forget={len(forget_rows)} retain={len(retain_rows)}")
    if len(forget_rows) == 0 or len(retain_rows) == 0:
        raise ValueError("Need both forget(label=1) and retain(label=0) samples for preference training.")

    forget_ds = PromptGenDataset(forget_rows)
    retain_ds = PromptGenDataset(retain_rows)

    forget_sampler = DistributedSampler(forget_ds, shuffle=True, drop_last=True)
    retain_sampler = DistributedSampler(retain_ds, shuffle=True, drop_last=True)

    collator = PromptGenCollator(tok, CollateCfg(max_length=args.max_length, add_eos=True))
    forget_loader = DataLoader(
        forget_ds,
        batch_size=args.batch_size,
        sampler=forget_sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=collator,
        drop_last=True,
    )
    retain_loader = DataLoader(
        retain_ds,
        batch_size=args.batch_size,
        sampler=retain_sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=collator,
        drop_last=True,
    )

    # load model: base then finetuned weights
    base_then_ft = load_base_then_finetuned(args.base_model, args.ckpt_dir, dtype=dtype)

    # reference model is frozen copy of the same starting point (base+finetuned)
    ref_model = load_base_then_finetuned(args.base_model, args.ckpt_dir, dtype=dtype)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()

    # mixed precision config for FSDP
    mp = None
    if args.fp16:
        mp = MixedPrecision(param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16)
    elif args.bf16:
        mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)

    # wrap in FSDP
    model = wrap_fsdp(base_then_ft, mp=mp, cpu_offload=args.cpu_offload)
    ref_model = wrap_fsdp(ref_model, mp=mp, cpu_offload=args.cpu_offload)

    # optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # estimate total steps
    steps_per_epoch = min(len(forget_loader), len(retain_loader)) // max(1, args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=max(total_steps, 1)
    )

    # maybe_resume_opt_sched(args.ckpt_dir, optimizer, scheduler, args.resume_optimizer)

    # wandb
    if is_main():
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=vars(args),
        )

    model.train()
    global_step = 0
    opt_step = 0

    # training loop
    pbar = tqdm(total=total_steps, disable=not is_main(), desc="NPO/FSDP opt steps")
    start_time = time.time()

    for epoch in range(args.epochs):
        forget_sampler.set_epoch(epoch)
        retain_sampler.set_epoch(epoch)

        # zip loaders (same length via min in steps calc); iterate explicitly
        forget_it = iter(forget_loader)
        retain_it = iter(retain_loader)

        # number of *micro* steps this epoch
        micro_steps = min(len(forget_loader), len(retain_loader))

        for ms in range(micro_steps):
            # fetch batches
            f_batch = next(forget_it)
            r_batch = next(retain_it)

            # move to GPU
            for k in f_batch:
                f_batch[k] = f_batch[k].cuda(non_blocking=True)
            for k in r_batch:
                r_batch[k] = r_batch[k].cuda(non_blocking=True)

            # forward current model
            out_r = model(
                input_ids=r_batch["input_ids"],
                attention_mask=r_batch["attention_mask"],
                labels=r_batch["labels"],
                use_cache=False,
            )
            out_f = model(
                input_ids=f_batch["input_ids"],
                attention_mask=f_batch["attention_mask"],
                labels=f_batch["labels"],
                use_cache=False,
            )

            pi_logp_r = sequence_logp_from_labels(out_r.logits, r_batch["labels"])
            pi_logp_f = sequence_logp_from_labels(out_f.logits, f_batch["labels"])
            tok_acc_r = masked_token_accuracy(out_r.logits[:, :-1, :], r_batch["labels"][:, 1:])
            tok_acc_f = masked_token_accuracy(out_f.logits[:, :-1, :], f_batch["labels"][:, 1:])

            # reference logp (no grad)
            with torch.no_grad():
                ref_out_r = ref_model(
                    input_ids=r_batch["input_ids"],
                    attention_mask=r_batch["attention_mask"],
                    labels=r_batch["labels"],
                    use_cache=False,
                )
                ref_out_f = ref_model(
                    input_ids=f_batch["input_ids"],
                    attention_mask=f_batch["attention_mask"],
                    labels=f_batch["labels"],
                    use_cache=False,
                )
                ref_logp_r = sequence_logp_from_labels(ref_out_r.logits, r_batch["labels"])
                ref_logp_f = sequence_logp_from_labels(ref_out_f.logits, f_batch["labels"])

            loss, pref_acc = npo_loss(
                pi_logp_chosen=pi_logp_r,
                pi_logp_rejected=pi_logp_f,
                ref_logp_chosen=ref_logp_r,
                ref_logp_rejected=ref_logp_f,
                beta=args.beta,
            )
            loss = loss / max(1, args.grad_accum)
            loss.backward()
            global_step += 1

            if global_step % args.grad_accum == 0:
                # grad norm before step
                gnorm = grad_global_norm(model)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                opt_step += 1
                pbar.update(1)

                if is_main() and (opt_step % args.log_every == 0):
                    wandb.log({
                        "loss": float(loss.item() * args.grad_accum),
                        "pref_accuracy": float(pref_acc.item()),
                        "tok_acc_retain": float(tok_acc_r.item()),
                        "tok_acc_forget": float(tok_acc_f.item()),
                        "pi_logp_retain_mean": float(pi_logp_r.mean().item()),
                        "pi_logp_forget_mean": float(pi_logp_f.mean().item()),
                        "ref_logp_retain_mean": float(ref_logp_r.mean().item()),
                        "ref_logp_forget_mean": float(ref_logp_f.mean().item()),
                        "grad_norm": float(gnorm),
                        "lr": float(scheduler.get_last_lr()[0]),
                        "opt_step": opt_step,
                        "epoch": epoch,
                        "elapsed_sec": time.time() - start_time,
                    }, step=opt_step)

                # periodic save
                if opt_step % args.save_every == 0:
                    save_full_model_fsdp(model, args.output_dir, tag=f"step{opt_step:06d}")
                    if is_main():
                        torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(args.output_dir, "scheduler.pt"))

                if args.max_steps > 0 and opt_step >= args.max_steps:
                    break

        if args.max_steps > 0 and opt_step >= args.max_steps:
            break

    pbar.close()

    # final save
    save_full_model_fsdp(model, args.output_dir, tag="final")
    if is_main():
        torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(args.output_dir, "scheduler.pt"))
        wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
