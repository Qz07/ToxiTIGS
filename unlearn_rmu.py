#!/usr/bin/env python3
"""
RMU unlearning for GPT-2 with FSDP (2 GPUs).

- Data: list[dict] with keys: prompt, generation, label
  * forget set: label==1
  * retain set: label==0
- Loss masking: supervised / RMU computed ONLY on generation tokens (prompt masked out)
- RMU losses (activation-based) using hidden states at chosen layer:
  L_forget  = MSE(h_updated, c*u) on forget gen tokens
  L_retain  = MSE(h_updated - h_frozen, 0) on retain gen tokens
  L_total   = L_forget + alpha * L_retain

- Loading: base model first (gpt2), then apply finetuned weights from checkpoint/pytorch_model.bin
- Optional resume: optimizer.pt, scheduler.pt

Run:
  torchrun --nproc_per_node=2 unlearn_rmu_fsdp.py \
    --data_path /path/to/data.pkl \
    --ckpt_dir  /path/to/train_lt_256/step_00000484 \
    --output_dir ./rmu_out \
    --wandb_project ToxiTIGS_RMU \
    --seq_len 256 --batch_size 2 --grad_accum 8 \
    --epochs 1 --lr 2e-5 --alpha 4.0 --c 1.0 --rmu_layer 8
"""

import argparse
import json
import math
import os
import pickle
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

# FSDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
from torch.distributed.fsdp import ShardingStrategy

try:
    import wandb
except Exception:
    wandb = None


# -------------------------
# Utils
# -------------------------

def is_rank0() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def setup_distributed():
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return x
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    y /= dist.get_world_size()
    return y


# -------------------------
# Data
# -------------------------

class ListDictDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def load_listdict(path: str) -> List[Dict[str, Any]]:
    # supports .pkl (pickle list), .json (list), .jsonl
    if path.endswith(".pkl") or path.endswith(".pickle"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, list):
            raise ValueError("Pickle must contain a list of dictionaries.")
        return obj

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, list):
            raise ValueError(".json must contain a list of dictionaries.")
        return obj

    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    raise ValueError("Unsupported data format. Use .pkl/.pickle, .json, or .jsonl")


@dataclass
class Batch:
    input_ids: torch.Tensor          # [B, T]
    attention_mask: torch.Tensor     # [B, T]
    gen_mask: torch.Tensor           # [B, T] bool; True where generation tokens are
    # for token-accuracy (optional)
    labels_next: torch.Tensor        # [B, T] int64; next-token labels (shifted), -100 masked


class CollatorGenMask:
    def __init__(self, tokenizer, seq_len: int):
        self.tok = tokenizer
        self.seq_len = seq_len

    def __call__(self, examples: List[Dict[str, Any]]) -> Batch:
        prompts = [ex["prompt"] for ex in examples]
        gens = [ex["generation"] for ex in examples]

        # tokenize prompt and generation separately so we can build a generation-only mask
        p = self.tok(prompts, add_special_tokens=False)
        g = self.tok(gens, add_special_tokens=False)

        input_ids = []
        attention_mask = []
        gen_mask = []
        labels_next = []

        eos = self.tok.eos_token_id

        for p_ids, g_ids in zip(p["input_ids"], g["input_ids"]):
            # sequence: prompt + generation + eos
            ids = (p_ids + g_ids + [eos])[: self.seq_len]
            attn = [1] * len(ids)

            # generation mask: True on generation tokens (+ eos) only
            # prompt part length may be truncated
            prompt_len = min(len(p_ids), self.seq_len)
            gen_start = prompt_len
            gm = [False] * len(ids)
            for t in range(gen_start, len(ids)):
                gm[t] = True

            # pad
            pad_id = self.tok.pad_token_id
            if pad_id is None:
                # GPT2 has no pad by default; set to eos in tokenizer setup
                pad_id = eos

            pad_n = self.seq_len - len(ids)
            if pad_n > 0:
                ids = ids + [pad_id] * pad_n
                attn = attn + [0] * pad_n
                gm = gm + [False] * pad_n

            ids_t = torch.tensor(ids, dtype=torch.long)
            attn_t = torch.tensor(attn, dtype=torch.long)
            gm_t = torch.tensor(gm, dtype=torch.bool)

            # next-token labels for *generation tokens* only (token accuracy logging)
            # labels_next[t] = input_ids[t+1], with last position ignored
            lbl = torch.full((self.seq_len,), -100, dtype=torch.long)
            # positions where we want to evaluate next-token prediction:
            # predict token at t+1 based on token at t, so mask at t where gm[t+1] is True.
            for t in range(self.seq_len - 1):
                if gm_t[t + 1] and attn_t[t + 1] == 1:
                    lbl[t] = ids_t[t + 1]
            # last token has no next token
            lbl[self.seq_len - 1] = -100

            input_ids.append(ids_t)
            attention_mask.append(attn_t)
            gen_mask.append(gm_t)
            labels_next.append(lbl)

        return Batch(
            input_ids=torch.stack(input_ids, dim=0),
            attention_mask=torch.stack(attention_mask, dim=0),
            gen_mask=torch.stack(gen_mask, dim=0),
            labels_next=torch.stack(labels_next, dim=0),
        )


# -------------------------
# RMU core
# -------------------------

def sample_u(hidden_size: int, device: torch.device, normalize: bool = True) -> torch.Tensor:
    # pseudocode: independent entries uniform[0,1)
    u = torch.rand(hidden_size, device=device)
    if normalize:
        u = u / (u.norm(p=2) + 1e-12)
    return u


def extract_layer_hidden(hidden_states: Tuple[torch.Tensor, ...], rmu_layer: int) -> torch.Tensor:
    """
    GPT2 returns hidden_states with length = n_layer + 1:
      hidden_states[0] is embeddings output
      hidden_states[i] is output after transformer block i-1
    So "layer k" block output is hidden_states[k+1].
    We'll accept rmu_layer in [0, n_layer-1] (transformer block index).
    """
    idx = rmu_layer + 1
    if idx < 0 or idx >= len(hidden_states):
        raise ValueError(f"rmu_layer={rmu_layer} out of range for hidden_states len={len(hidden_states)}")
    return hidden_states[idx]  # [B, T, H]


def masked_mse(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x, y: [B,T,H]
    mask: [B,T] bool (True positions used)
    """
    # avoid empty mask
    denom = mask.sum().clamp_min(1).to(x.dtype)
    diff2 = (x - y).pow(2).sum(dim=-1)  # [B,T]
    return (diff2 * mask.to(diff2.dtype)).sum() / denom


@torch.no_grad()
def token_accuracy_from_logits(logits: torch.Tensor, labels_next: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    logits: [B,T,V]
    labels_next: [B,T] with -100 masked
    returns: (correct_count, total_count) as tensors on same device
    """
    pred = torch.argmax(logits, dim=-1)  # [B,T]
    mask = labels_next != -100
    total = mask.sum()
    correct = ((pred == labels_next) & mask).sum()
    return correct, total


# -------------------------
# Main
# -------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data_path", type=str, required=True,
                   help="Path to .pkl/.json/.jsonl containing list of dicts with prompt,generation,label")
    p.add_argument("--ckpt_dir", type=str, required=True,
                   help="Directory containing your finetuned checkpoint files (pytorch_model.bin, optimizer.pt, scheduler.pt, meta.json)")
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--base_model", type=str, default="gpt2")
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)

    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_steps", type=int, default=-1, help="Override total steps; -1 means infer from data")

    # RMU hyperparams
    p.add_argument("--alpha", type=float, default=4.0, help="retain loss weight")
    p.add_argument("--c", type=float, default=1.0, help="target magnitude for forget activations: c*u")
    p.add_argument("--rmu_layer", type=int, default=8, help="GPT2 transformer block index [0..n_layer-1]")
    p.add_argument("--u_resample", type=str, default="step", choices=["step", "epoch", "never"],
                   help="When to resample u (random direction vector)")

    # Logging / saving
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--save_every", type=int, default=500)

    p.add_argument("--wandb_project", type=str, default="")
    p.add_argument("--wandb_run_name", type=str, default="")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    p.add_argument("--seed", type=int, default=42)

    # resume optimizer/scheduler from ckpt_dir
    p.add_argument("--resume_optimizer", action="store_true")
    p.add_argument("--resume_scheduler", action="store_true")

    # numerical / perf
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--fp16", action="store_true", help="Use fp16 mixed precision (recommended on A5000)")

    return p.parse_args()


def build_models(args, device: torch.device):
    """
    Load base model first, then apply finetuned weights from ckpt_dir/pytorch_model.bin.
    Also build a frozen reference model (not FSDP) for retain activation matching.
    """
    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token  # important for padding GPT2

    # base model
    base = AutoModelForCausalLM.from_pretrained(args.base_model)
    base.config.use_cache = False
    base.config.output_hidden_states = True

    # apply finetuned weights
    wpath = os.path.join(args.ckpt_dir, "pytorch_model.bin")
    if not os.path.exists(wpath):
        raise FileNotFoundError(f"Missing {wpath}")

    state = torch.load(wpath, map_location="cpu")
    missing, unexpected = base.load_state_dict(state, strict=False)
    if is_rank0():
        print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")
        if len(unexpected) > 0:
            print("  unexpected example:", unexpected[:10])

    # frozen ref model (same weights at start)
    frozen = AutoModelForCausalLM.from_pretrained(args.base_model)
    frozen.config.use_cache = False
    frozen.config.output_hidden_states = True
    frozen.load_state_dict(base.state_dict(), strict=True)
    frozen.to(device)
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)

    return tok, base, frozen


def wrap_fsdp(model: torch.nn.Module, fp16: bool) -> FSDP:
    from functools import partial
    from transformers.models.gpt2.modeling_gpt2 import GPT2Block
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
    from torch.distributed.fsdp import ShardingStrategy
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    mp = None
    if fp16:
        mp = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )

    # ✅ correct for your signature: transformer_auto_wrap_policy(module, recurse, nonwrapped_numel, transformer_layer_cls=...)
    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={GPT2Block})

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
    )
    return fsdp_model



@torch.no_grad()
def evaluate(model: FSDP, frozen: torch.nn.Module, loader: DataLoader, args, device) -> Dict[str, float]:
    model.eval()
    total_loss_f = torch.tensor(0.0, device=device)
    total_loss_r = torch.tensor(0.0, device=device)
    total_corr = torch.tensor(0, device=device, dtype=torch.long)
    total_cnt = torch.tensor(0, device=device, dtype=torch.long)

    # fixed u for eval for stability
    u = sample_u(model.module.config.n_embd, device=device, normalize=True)
    target = args.c * u  # [H]

    for batch in loader:
        input_ids = batch.input_ids.to(device, non_blocking=True)
        attention_mask = batch.attention_mask.to(device, non_blocking=True)
        gen_mask = batch.gen_mask.to(device, non_blocking=True)
        labels_next = batch.labels_next.to(device, non_blocking=True)

        out_u = model(input_ids=input_ids, attention_mask=attention_mask)
        hs_u = extract_layer_hidden(out_u.hidden_states, args.rmu_layer)  # [B,T,H]

        out_f = frozen(input_ids=input_ids, attention_mask=attention_mask)
        hs_f = extract_layer_hidden(out_f.hidden_states, args.rmu_layer)

        # forget-style loss (toward c*u) and retain-style loss (match frozen)
        tgt = target.view(1, 1, -1).expand_as(hs_u)
        lf = masked_mse(hs_u, tgt, gen_mask)
        lr = masked_mse(hs_u, hs_f, gen_mask)

        total_loss_f += lf
        total_loss_r += lr

        corr, cnt = token_accuracy_from_logits(out_u.logits, labels_next)
        total_corr += corr
        total_cnt += cnt

    # reduce across ranks
    total_loss_f = all_reduce_mean(total_loss_f)
    total_loss_r = all_reduce_mean(total_loss_r)
    total_corr = all_reduce_mean(total_corr.float()).long()  # ok since same size loaders recommended
    total_cnt = all_reduce_mean(total_cnt.float()).long()

    acc = (total_corr.float() / total_cnt.clamp_min(1).float()).item()
    return {
        "eval_forget_loss": total_loss_f.item() / max(len(loader), 1),
        "eval_retain_loss": total_loss_r.item() / max(len(loader), 1),
        "eval_token_acc": acc,
    }


def main():
    args = parse_args()
    setup_distributed()
    seed_all(args.seed + (dist.get_rank() if dist.is_initialized() else 0))

    device = torch.device("cuda", torch.cuda.current_device())

    os.makedirs(args.output_dir, exist_ok=True)
    if is_rank0():
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    # wandb
    use_wandb = (args.wandb_mode != "disabled") and (wandb is not None) and is_rank0() and (args.wandb_project != "")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name if args.wandb_run_name else None,
            entity=args.wandb_entity if args.wandb_entity else None,
            mode=args.wandb_mode,
            config=vars(args),
        )

    # build models
    tok, model, frozen = build_models(args, device)

    # FSDP wrap
    model = wrap_fsdp(model, fp16=args.fp16)
    model.train()

    # load data
    rows = load_listdict(args.data_path)
    forget_rows = [r for r in rows if int(r["label"]) == 1]
    retain_rows = [r for r in rows if int(r["label"]) == 0]
    if is_rank0():
        print(f"Loaded rows={len(rows)} forget={len(forget_rows)} retain={len(retain_rows)}")

    forget_ds = ListDictDataset(forget_rows)
    retain_ds = ListDictDataset(retain_rows)

    collate = CollatorGenMask(tok, seq_len=args.seq_len)

    # NOTE: simplest: each rank sees different shards via DistributedSampler (recommended),
    # but if you want strict determinism you can add it. Here we keep it simple:
    forget_loader = DataLoader(forget_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    retain_loader = DataLoader(retain_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True)

    # optimizer / scheduler
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # total steps
    steps_per_epoch = min(len(forget_loader), len(retain_loader))
    if steps_per_epoch == 0:
        raise ValueError("Forget/retain loaders are empty. Check labels and data.")

    total_steps = steps_per_epoch * args.epochs
    if args.max_steps > 0:
        total_steps = args.max_steps

    warmup_steps = int(total_steps * args.warmup_ratio)
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # optional resume
    if args.resume_optimizer:
        opath = os.path.join(args.ckpt_dir, "optimizer.pt")
        if os.path.exists(opath):
            ck = torch.load(opath, map_location="cpu")
            optim.load_state_dict(ck)
            if is_rank0():
                print(f"Resumed optimizer from {opath}")
        else:
            if is_rank0():
                print(f"[warn] resume_optimizer set but missing {opath}")

    if args.resume_scheduler:
        spath = os.path.join(args.ckpt_dir, "scheduler.pt")
        if os.path.exists(spath):
            ck = torch.load(spath, map_location="cpu")
            sched.load_state_dict(ck)
            if is_rank0():
                print(f"Resumed scheduler from {spath}")
        else:
            if is_rank0():
                print(f"[warn] resume_scheduler set but missing {spath}")

    # training
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    hidden_size = model.module.config.n_embd
    u = sample_u(hidden_size, device=device, normalize=True)  # initial u
    global_step = 0

    pbar = tqdm(total=total_steps, disable=not is_rank0(), dynamic_ncols=True, desc="RMU train")

    retain_iter = iter(retain_loader)
    forget_iter = iter(forget_loader)

    def next_batch(it, loader):
        try:
            return next(it), it
        except StopIteration:
            it = iter(loader)
            return next(it), it

    while global_step < total_steps:
        # u resampling
        if args.u_resample == "step":
            u = sample_u(hidden_size, device=device, normalize=True)
        elif args.u_resample == "epoch":
            # resample when at epoch boundary (approx)
            if global_step % steps_per_epoch == 0:
                u = sample_u(hidden_size, device=device, normalize=True)
        # else: never

        # fetch forget/retain batches
        b_forget, forget_iter = next_batch(forget_iter, forget_loader)
        b_retain, retain_iter = next_batch(retain_iter, retain_loader)

        # move to device
        f_input_ids = b_forget.input_ids.to(device, non_blocking=True)
        f_attn = b_forget.attention_mask.to(device, non_blocking=True)
        f_gmask = b_forget.gen_mask.to(device, non_blocking=True)

        r_input_ids = b_retain.input_ids.to(device, non_blocking=True)
        r_attn = b_retain.attention_mask.to(device, non_blocking=True)
        r_gmask = b_retain.gen_mask.to(device, non_blocking=True)
        r_labels_next = b_retain.labels_next.to(device, non_blocking=True)

        # grad accum
        optim.zero_grad(set_to_none=True)

        accum_forget = 0.0
        accum_retain = 0.0
        accum_acc = 0.0
        accum_cnt = 0.0

        for micro in range(args.grad_accum):
            # reuse same batches across micros (simple); if you prefer, you can draw new batches each micro
            with torch.cuda.amp.autocast(enabled=args.fp16):
                # updated model forward on forget
                out_f_u = model(input_ids=f_input_ids, attention_mask=f_attn)
                hs_f_u = extract_layer_hidden(out_f_u.hidden_states, args.rmu_layer)  # [B,T,H]
                tgt = (args.c * u).view(1, 1, -1).expand_as(hs_f_u)
                loss_forget = masked_mse(hs_f_u, tgt, f_gmask)

                # updated model forward on retain
                out_r_u = model(input_ids=r_input_ids, attention_mask=r_attn)
                hs_r_u = extract_layer_hidden(out_r_u.hidden_states, args.rmu_layer)

                # frozen forward on retain
                with torch.no_grad():
                    out_r_f = frozen(input_ids=r_input_ids, attention_mask=r_attn)
                    hs_r_f = extract_layer_hidden(out_r_f.hidden_states, args.rmu_layer)

                loss_retain = masked_mse(hs_r_u, hs_r_f, r_gmask)

                loss = (loss_forget + args.alpha * loss_retain) / args.grad_accum

            scaler.scale(loss).backward()

            accum_forget += float(loss_forget.detach())
            accum_retain += float(loss_retain.detach())

            # token accuracy on retain generation tokens (next-token)
            with torch.no_grad():
                corr, cnt = token_accuracy_from_logits(out_r_u.logits, r_labels_next)
                accum_acc += float(corr)
                accum_cnt += float(cnt)

        # unscale + clip + step
        scaler.unscale_(optim)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optim)
        scaler.update()
        sched.step()

        global_step += 1
        pbar.update(1)

        # logging
        if is_rank0() and (global_step % args.log_every == 0 or global_step == 1):
            lr = sched.get_last_lr()[0]
            tok_acc = (accum_acc / max(accum_cnt, 1.0))

            log = {
                "step": global_step,
                "lr": lr,
                "forget_loss": accum_forget / args.grad_accum,
                "retain_loss": accum_retain / args.grad_accum,
                "total_loss": (accum_forget / args.grad_accum) + args.alpha * (accum_retain / args.grad_accum),
                "grad_norm": float(grad_norm.detach().cpu()),
                "retain_token_acc": tok_acc,
            }
            pbar.set_postfix({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in log.items() if k != "step"})
            if use_wandb:
                wandb.log(log, step=global_step)

        # lightweight eval (retain set) + save
        if global_step % args.eval_every == 0:
            # small eval loader sample to keep it cheap
            # (you can replace with a proper DistributedSampler + full pass)
            eval_loader = DataLoader(retain_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, drop_last=False)
            metrics = evaluate(model, frozen, eval_loader, args, device)
            if is_rank0():
                print("\nEVAL:", metrics)
                if use_wandb:
                    wandb.log(metrics, step=global_step)

            model.train()

        if global_step % args.save_every == 0 or global_step == total_steps:
            if is_rank0():
                save_dir = os.path.join(args.output_dir, f"step_{global_step:08d}")
                os.makedirs(save_dir, exist_ok=True)

                # save model weights (gathered by rank0)
                # WARNING: this saves *full* state dict on rank0. OK for GPT2.
                full_state = model.state_dict()
                torch.save(full_state, os.path.join(save_dir, "pytorch_model.bin"))

                # save optimizer / scheduler
                torch.save(optim.state_dict(), os.path.join(save_dir, "optimizer.pt"))
                torch.save(sched.state_dict(), os.path.join(save_dir, "scheduler.pt"))

                meta = {
                    "global_step": global_step,
                    "base_model": args.base_model,
                    "ckpt_dir_applied": args.ckpt_dir,
                    "rmu_layer": args.rmu_layer,
                    "alpha": args.alpha,
                    "c": args.c,
                }
                with open(os.path.join(save_dir, "meta.json"), "w") as f:
                    json.dump(meta, f, indent=2)

                print(f"\nSaved checkpoint to {save_dir}")

    pbar.close()
    if use_wandb:
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    main()
