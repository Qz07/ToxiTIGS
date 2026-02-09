#!/usr/bin/env python3
"""
FSDP IdkDPO for GPT-2

Goal:
  - For label=1 ("toxic / forget") prompts: align model to output an "I don't know" response
    using a DPO-style objective (chosen = IDK, rejected = toxic generation from data).
  - For label=0 ("retain") prompts: train normally with next-token SFT on the provided generation
    (prompt tokens masked in labels, supervise generation tokens only).

Data format: list[dict] with keys:
  - prompt: str
  - generation: str
  - label: int (1=forget/toxic, 0=retain)

Core stability fixes included:
  - Token-level concat (prompt_ids + gen_ids) => exact prompt/gen boundary
  - Length-based attention_mask (works even when pad_token_id == eos_token_id)
  - DPO uses mean logp per supervised token (length-normalized)
  - bf16 recommended; fp16 uses AMP GradScaler
  - Grad clipping + optional FSDP no_sync for grad accumulation

Run:
  torchrun --nproc_per_node=2 train_idkdpo_fsdp.py \
    --data_path data.jsonl --ckpt_dir /path/to/ckpt --output_dir /path/to/out \
    --bf16 --beta 0.1 --dpo_coef 1.0 --retain_coef 1.0 --use_no_sync
"""

import os
import json
import math
import time
import pickle
import random
import argparse
import contextlib
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

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

def validate_row(r: Dict[str, Any]) -> Dict[str, Any]:
    for k in ["prompt", "generation", "label"]:
        if k not in r:
            raise ValueError(f"Missing key '{k}' in row: {r}")
    prompt = r["prompt"]
    gen = r["generation"]
    lab = int(r["label"])
    if not isinstance(prompt, str) or not isinstance(gen, str):
        raise ValueError(f"prompt/generation must be str, got {type(prompt)} / {type(gen)}")
    if lab not in (0, 1):
        raise ValueError(f"label must be 0 or 1, got {lab}")
    return {"prompt": prompt, "generation": gen, "label": lab}

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


# ----------------------------
# Tokenization helpers (exact boundary + correct attention mask)
# ----------------------------

@dataclass
class CollateCfg:
    max_length: int
    add_eos: bool = True

class TokenConcatHelper:
    """
    Builds:
      input_ids = prompt_ids + gen_ids (+ eos)
      labels = input_ids but prompt positions masked to -100
      attention_mask = length-based (robust when pad_id == eos_id)
    """
    def __init__(self, tokenizer: AutoTokenizer, cfg: CollateCfg):
        self.tok = tokenizer
        self.cfg = cfg

    def encode_concat(self, prompt: str, gen: str) -> Tuple[torch.Tensor, torch.Tensor, int]:
        prompt_ids = self.tok(prompt, add_special_tokens=False, truncation=False)["input_ids"]
        gen_ids = self.tok(gen, add_special_tokens=False, truncation=False)["input_ids"]

        max_len = self.cfg.max_length
        eos_extra = 1 if self.cfg.add_eos else 0

        # truncate prompt first if needed
        if len(prompt_ids) > max_len - eos_extra:
            prompt_ids = prompt_ids[: max_len - eos_extra]
            gen_ids = []
        else:
            allowed_gen = (max_len - eos_extra) - len(prompt_ids)
            if allowed_gen < len(gen_ids):
                gen_ids = gen_ids[:allowed_gen]

        full_ids = prompt_ids + gen_ids
        if self.cfg.add_eos:
            full_ids = full_ids + [self.tok.eos_token_id]

        labels = full_ids.copy()
        prm_len = len(prompt_ids)
        for i in range(min(prm_len, len(labels))):
            labels[i] = -100

        length = len(full_ids)
        return torch.tensor(full_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long), length

    @staticmethod
    def length_mask(lengths: List[int], max_len: int) -> torch.Tensor:
        ar = torch.arange(max_len, dtype=torch.long).unsqueeze(0)  # [1, T]
        ln = torch.tensor(lengths, dtype=torch.long).unsqueeze(1)  # [B, 1]
        return (ar < ln).long()  # [B, T]


# ----------------------------
# Datasets
# ----------------------------

class RetainSFTDataset(Dataset):
    """
    label=0 rows: standard next-token SFT on provided generation
    """
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        return {"prompt": r["prompt"], "generation": r["generation"]}

class ForgetIdkDPODataset(Dataset):
    """
    label=1 rows: DPO preference
      chosen   = idk_text
      rejected = toxic generation from data
    """
    def __init__(self, rows: List[Dict[str, Any]], idk_text: str):
        self.rows = rows
        self.idk_text = idk_text

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        return {"prompt": r["prompt"], "rejected_generation": r["generation"], "chosen_generation": self.idk_text}


# ----------------------------
# Collators
# ----------------------------

class RetainSFTCollator:
    def __init__(self, tok: AutoTokenizer, cfg: CollateCfg):
        self.helper = TokenConcatHelper(tok, cfg)
        self.tok = tok

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        ids_list, labels_list, lens = [], [], []
        for ex in batch:
            ids, labels, ln = self.helper.encode_concat(ex["prompt"], ex["generation"])
            ids_list.append(ids); labels_list.append(labels); lens.append(ln)

        pad_id = self.tok.pad_token_id
        input_ids = nn.utils.rnn.pad_sequence(ids_list, batch_first=True, padding_value=pad_id)
        labels = nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
        attention_mask = self.helper.length_mask(lens, input_ids.size(1))

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

class ForgetDPOCollator:
    def __init__(self, tok: AutoTokenizer, cfg: CollateCfg):
        self.helper = TokenConcatHelper(tok, cfg)
        self.tok = tok

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        c_ids_list, c_lab_list, c_lens = [], [], []
        r_ids_list, r_lab_list, r_lens = [], [], []

        for ex in batch:
            c_ids, c_lab, c_ln = self.helper.encode_concat(ex["prompt"], ex["chosen_generation"])
            r_ids, r_lab, r_ln = self.helper.encode_concat(ex["prompt"], ex["rejected_generation"])
            c_ids_list.append(c_ids); c_lab_list.append(c_lab); c_lens.append(c_ln)
            r_ids_list.append(r_ids); r_lab_list.append(r_lab); r_lens.append(r_ln)

        pad_id = self.tok.pad_token_id
        chosen_input_ids = nn.utils.rnn.pad_sequence(c_ids_list, batch_first=True, padding_value=pad_id)
        chosen_labels = nn.utils.rnn.pad_sequence(c_lab_list, batch_first=True, padding_value=-100)
        rejected_input_ids = nn.utils.rnn.pad_sequence(r_ids_list, batch_first=True, padding_value=pad_id)
        rejected_labels = nn.utils.rnn.pad_sequence(r_lab_list, batch_first=True, padding_value=-100)

        chosen_attention_mask = self.helper.length_mask(c_lens, chosen_input_ids.size(1))
        rejected_attention_mask = self.helper.length_mask(r_lens, rejected_input_ids.size(1))

        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
            "rejected_labels": rejected_labels,
        }


# ----------------------------
# DPO objective (length-normalized logp)
# ----------------------------

def sequence_logp_mean_from_labels(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mean log-prob over positions where labels != -100.
    logits: [B, T, V], labels: [B, T]
    returns:
      - mean_logp: [B]
      - token_count: [B] (# supervised tokens)
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    log_probs = torch.log_softmax(shift_logits, dim=-1)  # [B, T-1, V]
    mask = shift_labels != -100
    gather_labels = shift_labels.masked_fill(~mask, 0).unsqueeze(-1)  # [B, T-1, 1]
    token_logp = log_probs.gather(-1, gather_labels).squeeze(-1)      # [B, T-1]
    token_logp = token_logp * mask.float()

    token_count = mask.sum(dim=-1).clamp_min(1).float()
    mean_logp = token_logp.sum(dim=-1) / token_count
    return mean_logp, token_count

def dpo_loss(
    pi_logp_chosen: torch.Tensor,
    pi_logp_rejected: torch.Tensor,
    ref_logp_chosen: torch.Tensor,
    ref_logp_rejected: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    logits = beta * [ (pi - ref)_chosen - (pi - ref)_rejected ]
    loss = -log(sigmoid(logits))
    """
    logits = beta * ((pi_logp_chosen - ref_logp_chosen) - (pi_logp_rejected - ref_logp_rejected))
    loss = -torch.nn.functional.logsigmoid(logits).mean()
    pref_acc = (logits > 0).float().mean()
    return loss, pref_acc


# ----------------------------
# FSDP wrapping + checkpointing
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

def load_base_then_finetuned(base_model: str, ckpt_dir: str, dtype: torch.dtype, ft_filename: str) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    ft_path = os.path.join(ckpt_dir, ft_filename)
    if not os.path.exists(ft_path):
        raise FileNotFoundError(f"Missing finetuned weights: {ft_path}")

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

def save_full_model_fsdp(model: FSDP, out_dir: str, filename: str):
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, filename)

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

    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--ckpt_dir", type=str, required=True)
    p.add_argument("--ft_filename", type=str, default="pytorch_model.bin")
    p.add_argument("--base_model", type=str, default="gpt2")
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--batch_size_retain", type=int, default=2, help="Per-GPU retain SFT batch size")
    p.add_argument("--batch_size_forget", type=int, default=2, help="Per-GPU forget DPO batch size")
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=-1, help="If >0, stop after this many optimizer steps")

    # IdkDPO specifics
    p.add_argument("--idk_text", type=str, default=" I don't know.",
                   help="Chosen response for label=1 prompts. Leading space is intentional for GPT-2 BPE.")
    p.add_argument("--beta", type=float, default=0.1, help="DPO beta (preference strength)")
    p.add_argument("--dpo_coef", type=float, default=1.0, help="Weight for DPO loss on forget prompts")
    p.add_argument("--retain_coef", type=float, default=1.0, help="Weight for SFT loss on retain prompts")
    p.add_argument("--idk_lm_coef", type=float, default=0.0,
                   help="Optional extra LM loss on IDK chosen sequence (helps lock in the phrasing).")

    # precision
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--cpu_offload", action="store_true")

    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--use_no_sync", action="store_true")

    # logging/saving
    p.add_argument("--wandb_project", type=str, default="idkdpo-unlearning")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--resume_optimizer", action="store_true")

    return p.parse_args()

def main():
    args = parse_args()
    setup_distributed()
    rank = dist.get_rank() if dist.is_initialized() else 0
    set_seed(args.seed + rank)

    # dtype
    if args.bf16 and args.fp16:
        raise ValueError("Choose only one: --bf16 or --fp16")
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token  # ok because we use length-based masks

    # load data
    rows = [validate_row(r) for r in read_list_of_dicts(args.data_path)]
    forget_rows = [r for r in rows if r["label"] == 1]
    retain_rows = [r for r in rows if r["label"] == 0]

    if is_main():
        print(f"[data] total={len(rows)} forget(label=1)={len(forget_rows)} retain(label=0)={len(retain_rows)}")

    if len(forget_rows) == 0:
        raise ValueError("No label=1 (forget/toxic) rows found. IdkDPO needs toxic prompts.")
    if len(retain_rows) == 0:
        raise ValueError("No label=0 (retain) rows found. Retain SFT needed to preserve fluency/utility.")

    # datasets/loaders
    forget_ds = ForgetIdkDPODataset(forget_rows, idk_text=args.idk_text)
    retain_ds = RetainSFTDataset(retain_rows)

    forget_sampler = DistributedSampler(forget_ds, shuffle=True, drop_last=True)
    retain_sampler = DistributedSampler(retain_ds, shuffle=True, drop_last=True)

    cfg = CollateCfg(max_length=args.max_length, add_eos=True)
    forget_collator = ForgetDPOCollator(tok, cfg)
    retain_collator = RetainSFTCollator(tok, cfg)

    forget_loader = DataLoader(
        forget_ds,
        batch_size=args.batch_size_forget,
        sampler=forget_sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=forget_collator,
        drop_last=True,
    )
    retain_loader = DataLoader(
        retain_ds,
        batch_size=args.batch_size_retain,
        sampler=retain_sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=retain_collator,
        drop_last=True,
    )

    # models (pi + frozen reference)
    pi_model = load_base_then_finetuned(args.base_model, args.ckpt_dir, dtype=dtype, ft_filename=args.ft_filename)
    ref_model = load_base_then_finetuned(args.base_model, args.ckpt_dir, dtype=dtype, ft_filename=args.ft_filename)
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # FSDP MP config
    mp = None
    if args.fp16:
        mp = MixedPrecision(param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16)
    elif args.bf16:
        mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)

    model = wrap_fsdp(pi_model, mp=mp, cpu_offload=args.cpu_offload)
    ref_model = wrap_fsdp(ref_model, mp=mp, cpu_offload=args.cpu_offload)
    ref_model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # steps: cycle the smaller loader so both losses keep training
    steps_per_epoch = max(len(forget_loader), len(retain_loader)) // max(1, args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=max(total_steps, 1)
    )

    maybe_resume_opt_sched(args.ckpt_dir, optimizer, scheduler, args.resume_optimizer)

    # wandb
    if is_main():
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=vars(args),
        )

    # AMP scaler for fp16
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    def autocast_ctx():
        if args.fp16:
            return torch.cuda.amp.autocast(dtype=torch.float16)
        if args.bf16:
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)
        return torch.cuda.amp.autocast(enabled=False)

    model.train()
    pbar = tqdm(total=total_steps, disable=not is_main(), desc="IdkDPO/FSDP opt steps")
    start_time = time.time()

    forget_it = iter(forget_loader)
    retain_it = iter(retain_loader)

    micro = 0
    opt_step = 0

    for epoch in range(args.epochs):
        forget_sampler.set_epoch(epoch)
        retain_sampler.set_epoch(epoch)

        # reset iterators each epoch
        forget_it = iter(forget_loader)
        retain_it = iter(retain_loader)

        # number of microsteps we want this epoch:
        micro_steps = max(len(forget_loader), len(retain_loader))
        for _ in range(micro_steps):
            # cycle loaders
            try:
                f_batch = next(forget_it)
            except StopIteration:
                forget_it = iter(forget_loader)
                f_batch = next(forget_it)

            try:
                r_batch = next(retain_it)
            except StopIteration:
                retain_it = iter(retain_loader)
                r_batch = next(retain_it)

            # move to GPU
            for k in f_batch:
                f_batch[k] = f_batch[k].cuda(non_blocking=True)
            for k in r_batch:
                r_batch[k] = r_batch[k].cuda(non_blocking=True)

            micro_in_accum = (micro % args.grad_accum) + 1
            do_step = (micro_in_accum == args.grad_accum)

            sync_ctx = model.no_sync() if (args.use_no_sync and not do_step) else contextlib.nullcontext()

            with sync_ctx:
                with autocast_ctx():
                    # ---- Retain SFT ----
                    out_ret = model(
                        input_ids=r_batch["input_ids"],
                        attention_mask=r_batch["attention_mask"],
                        labels=r_batch["labels"],
                        use_cache=False,
                    )
                    retain_loss = out_ret.loss  # mean CE over supervised (generation) tokens
                    retain_tok_acc = masked_token_accuracy(out_ret.logits[:, :-1, :], r_batch["labels"][:, 1:])

                    # ---- Forget DPO (IDK preferred) ----
                    out_c = model(
                        input_ids=f_batch["chosen_input_ids"],
                        attention_mask=f_batch["chosen_attention_mask"],
                        labels=f_batch["chosen_labels"],
                        use_cache=False,
                    )
                    out_rej = model(
                        input_ids=f_batch["rejected_input_ids"],
                        attention_mask=f_batch["rejected_attention_mask"],
                        labels=f_batch["rejected_labels"],
                        use_cache=False,
                    )

                    pi_logp_c, c_tokcnt = sequence_logp_mean_from_labels(out_c.logits, f_batch["chosen_labels"])
                    pi_logp_rj, rj_tokcnt = sequence_logp_mean_from_labels(out_rej.logits, f_batch["rejected_labels"])

                    tok_acc_c = masked_token_accuracy(out_c.logits[:, :-1, :], f_batch["chosen_labels"][:, 1:])
                    tok_acc_rj = masked_token_accuracy(out_rej.logits[:, :-1, :], f_batch["rejected_labels"][:, 1:])

                    with torch.no_grad():
                        ref_out_c = ref_model(
                            input_ids=f_batch["chosen_input_ids"],
                            attention_mask=f_batch["chosen_attention_mask"],
                            labels=f_batch["chosen_labels"],
                            use_cache=False,
                        )
                        ref_out_rj = ref_model(
                            input_ids=f_batch["rejected_input_ids"],
                            attention_mask=f_batch["rejected_attention_mask"],
                            labels=f_batch["rejected_labels"],
                            use_cache=False,
                        )
                        ref_logp_c, _ = sequence_logp_mean_from_labels(ref_out_c.logits, f_batch["chosen_labels"])
                        ref_logp_rj, _ = sequence_logp_mean_from_labels(ref_out_rj.logits, f_batch["rejected_labels"])

                    dpo, pref_acc = dpo_loss(
                        pi_logp_chosen=pi_logp_c,
                        pi_logp_rejected=pi_logp_rj,
                        ref_logp_chosen=ref_logp_c,
                        ref_logp_rejected=ref_logp_rj,
                        beta=args.beta,
                    )

                    # Optional extra LM loss on the chosen IDK completion (helps lock in exact phrasing)
                    idk_lm_loss = out_c.loss if out_c.loss is not None else torch.tensor(0.0, device=dpo.device)

                    # combined loss
                    loss = (args.retain_coef * retain_loss) + (args.dpo_coef * dpo) + (args.idk_lm_coef * idk_lm_loss)
                    loss = loss / max(1, args.grad_accum)

                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            micro += 1

            if do_step:
                if args.fp16:
                    scaler.unscale_(optimizer)

                # clip grads (FSDP-aware)
                if args.grad_clip > 0:
                    try:
                        _ = FSDP.clip_grad_norm_(model.parameters(), args.grad_clip)
                    except Exception:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                opt_step += 1
                pbar.update(1)

                if is_main() and (opt_step % args.log_every == 0):
                    wandb.log({
                        "loss_total": float(loss.item() * args.grad_accum),
                        "loss_retain_sft": float(retain_loss.item()),
                        "loss_dpo_forget": float(dpo.item()),
                        "loss_idk_lm": float(idk_lm_loss.item()),
                        "pref_accuracy": float(pref_acc.item()),
                        "tok_acc_retain": float(retain_tok_acc.item()),
                        "tok_acc_idk_chosen": float(tok_acc_c.item()),
                        "tok_acc_forget_rejected": float(tok_acc_rj.item()),
                        "pi_logp_idk_mean": float(pi_logp_c.mean().item()),
                        "pi_logp_rejected_mean": float(pi_logp_rj.mean().item()),
                        "ref_logp_idk_mean": float(ref_logp_c.mean().item()),
                        "ref_logp_rejected_mean": float(ref_logp_rj.mean().item()),
                        "idk_sup_tokens_mean": float(c_tokcnt.mean().item()),
                        "rej_sup_tokens_mean": float(rj_tokcnt.mean().item()),
                        "lr": float(scheduler.get_last_lr()[0]),
                        "opt_step": opt_step,
                        "epoch": epoch,
                        "elapsed_sec": time.time() - start_time,
                    }, step=opt_step)

                if opt_step % args.save_every == 0:
                    save_full_model_fsdp(model, args.output_dir, filename=f"pytorch_model_step{opt_step:06d}.bin")
                    if is_main():
                        torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(args.output_dir, "scheduler.pt"))

                if args.max_steps > 0 and opt_step >= args.max_steps:
                    break

        if args.max_steps > 0 and opt_step >= args.max_steps:
            break

    pbar.close()

    # final save (HF-default filename for convenience)
    save_full_model_fsdp(model, args.output_dir, filename="pytorch_model.bin")
    if is_main():
        torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(args.output_dir, "scheduler.pt"))
        wandb.finish()

    cleanup_distributed()

if __name__ == "__main__":
    main()
