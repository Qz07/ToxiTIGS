#!/usr/bin/env python3
# unlearn_rmu_gpt2_fsdp.py
"""
RMU unlearning for GPT-2 (causal LM / NTP) with:
- dataset: pickle file containing a list[dict] with keys: 'prompt', 'generation', 'label'
    - label==1 => forget set
    - label==0 => retain set
- supervision ONLY on generation tokens (prompt tokens masked with -100)
- RMU objective (activation rewriting) on a sampled layer each epoch:
    L = L_forget_rmu + alpha * L_retain_match
- Optional: add GA on forget set (gradient ascent on LM loss) via --ga_forget_weight > 0
- FSDP across 2 GPUs (torchrun)
- wandb logging (rank0 only)
- resume from your step dir that contains:
    meta.json, pytorch_model.bin, optimizer.pt, scheduler.pt

Run example:
  torchrun --nproc_per_node=2 unlearn_rmu_gpt2_fsdp.py \
    --data_path /path/to/data.pkl \
    --model_name_or_path gpt2 \
    --resume_dir /path/to/train_lt_256/step_00000484 \
    --output_dir /path/to/rmu_out \
    --seq_len 256 --batch_size 2 --grad_accum 8 --epochs 4 \
    --lr 3e-4 --weight_decay 1e-4 \
    --alpha 1000 --c 4.0 --k_schedule 0.75,1.0,1.0,1.0 \
    --wandb_project toxic-unlearning --run_name rmu-gpt2-fsdp
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass
from itertools import cycle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# FSDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


# -------------------------
# Utils
# -------------------------
def is_rank0() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def ddp_setup(local_rank: int):
    if dist.is_initialized():
        return
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)


def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


def load_json_if_exists(path: str) -> Optional[dict]:
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def torch_load_if_exists(path: str, map_location="cpu"):
    if path and os.path.exists(path):
        return torch.load(path, map_location=map_location)
    return None


def parse_k_schedule(k_schedule_str: str, epochs: int) -> List[float]:
    if not k_schedule_str:
        ks = [0.75] + [1.0] * max(epochs - 1, 0)
        return ks[:epochs]
    parts = [float(x.strip()) for x in k_schedule_str.split(",") if x.strip()]
    if not parts:
        parts = [1.0]
    if len(parts) < epochs:
        parts = parts + [parts[-1]] * (epochs - len(parts))
    return parts[:epochs]


def get_layer_range_indices(k: float, num_layers: int) -> Tuple[int, int]:
    if k == 0.25:
        start_idx = 0
        end_idx = int(num_layers * 0.25) - 1
    elif k == 0.50:
        start_idx = int(num_layers * 0.25)
        end_idx = int(num_layers * 0.50) - 1
    elif k == 0.75:
        start_idx = int(num_layers * 0.50)
        end_idx = int(num_layers * 0.75) - 1
    elif k == 1.0:
        start_idx = int(num_layers * 0.75)
        end_idx = num_layers - 1
    else:
        raise ValueError(f"Unknown k value: {k}")
    start_idx = max(0, min(start_idx, num_layers - 1))
    end_idx = max(start_idx, min(end_idx, num_layers - 1))
    return start_idx, end_idx


def pair_full_epoch(forget_loader, retain_loader):
    len_f, len_r = len(forget_loader), len(retain_loader)
    if len_f >= len_r:
        main_iter = zip(forget_loader, cycle(retain_loader))
        num_steps = len_f
    else:
        main_iter = zip(cycle(forget_loader), retain_loader)
        num_steps = len_r
    return main_iter, num_steps


def linear_alpha(epoch: int, epochs: int, alpha: float, alpha_start: float, alpha_end: float) -> float:
    if alpha_start >= 0 and alpha_end >= 0:
        if epochs <= 1:
            return alpha_end
        t = epoch / (epochs - 1)
        return alpha_start + t * (alpha_end - alpha_start)
    return alpha


def all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return x
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    y /= dist.get_world_size()
    return y


# -------------------------
# Dataset / Collator
# -------------------------
class PromptGenDataset(Dataset):
    def __init__(self, data_list: List[dict]):
        self.data = data_list

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        ex = self.data[idx]
        return {
            "prompt": ex["prompt"],
            "generation": ex["generation"],
            "label": int(ex["label"]),
        }


@dataclass
class CausalCollator:
    tokenizer: any
    seq_len: int

    def __call__(self, batch: List[dict]) -> Dict[str, torch.Tensor]:
        prompts = [b["prompt"] for b in batch]
        gens = [b["generation"] for b in batch]
        labels01 = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)

        # tokenize prompt alone to get prompt length
        prompt_tok = self.tokenizer(
            prompts,
            add_special_tokens=False,
            truncation=True,
            max_length=self.seq_len,
            return_attention_mask=False,
        )
        prompt_lens = [len(x) for x in prompt_tok["input_ids"]]

        # tokenize full text = prompt + generation
        full_text = [p + g for p, g in zip(prompts, gens)]
        full = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.seq_len,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = full["input_ids"]
        attn = full["attention_mask"]

        # labels for causal LM: same shape as input_ids; mask prompt tokens
        lm_labels = input_ids.clone()
        for i, pl in enumerate(prompt_lens):
            pl = min(pl, self.seq_len)
            lm_labels[i, :pl] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": lm_labels,
            "label01": labels01,  # not used by model; just for debugging
        }


# -------------------------
# RMU layer selection for GPT-2
# -------------------------
def get_gpt2_rmu_layers(model: torch.nn.Module) -> List[str]:
    """
    Return a list of layer module names whose activations we can hook.
    We pick projection outputs inside each block (stable hidden size):
      - transformer.h.{i}.attn.c_proj
      - transformer.h.{i}.mlp.c_proj

    Order matters (earlier -> later).
    """
    names: List[str] = []
    # Works for GPT2LMHeadModel: model.transformer.h is a ModuleList
    n_blocks = len(model.transformer.h)
    for i in range(n_blocks):
        attn_name = f"transformer.h.{i}.attn.c_proj"
        mlp_name = f"transformer.h.{i}.mlp.c_proj"
        # Only include if present (robustness)
        try:
            _ = model.get_submodule(attn_name)
            names.append(attn_name)
        except Exception:
            pass
        try:
            _ = model.get_submodule(mlp_name)
            names.append(mlp_name)
        except Exception:
            pass
    if len(names) == 0:
        raise RuntimeError("No RMU-hookable GPT-2 layers found. Check module names.")
    return names


def create_random_u_vectors(layer_names: List[str], hidden_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    For GPT-2, hooked activations are typically (B, T, C) where C=hidden_size.
    So each u is a (1, 1, C) unit vector.
    """
    u = {}
    for ln in layer_names:
        v = torch.rand(hidden_size, device=device)
        v = v / (torch.norm(v) + 1e-12)
        u[ln] = v.view(1, 1, -1)
    return u


def compute_layer_rms_gpt2(
    model_frozen: torch.nn.Module,
    layer_name: str,
    retain_loader,
    device: torch.device,
    num_batches: int = 2,
) -> float:
    activations: Dict[str, torch.Tensor] = {}

    def hook_fn(_m, _inp, out):
        # out can be (B,T,C)
        activations["act"] = out

    handle = model_frozen.get_submodule(layer_name).register_forward_hook(hook_fn)
    model_frozen.eval()
    cnt = 0
    with torch.no_grad():
        for batch in retain_loader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            _ = model_frozen(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            if "act" in activations:
                act = activations["act"].detach()
                rms = torch.sqrt(torch.mean(act.float() ** 2)).item()
                handle.remove()
                return float(rms)
            cnt += 1
            if cnt >= num_batches:
                break
    handle.remove()
    return 1.0


# -------------------------
# Checkpoint IO
# -------------------------
def _is_hf_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json"))


def load_model_with_optional_step_checkpoint(
    model_name_or_path: str,
    resume_dir: Optional[str],
    torch_dtype: torch.dtype,
) -> Tuple[torch.nn.Module, any, dict]:
    """
    Loads GPT-2 base model, then (optionally) loads a step checkpoint that only has
    pytorch_model.bin (+ optimizer/scheduler/meta).
    Returns (model, tokenizer, meta_dict).
    """
    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cfg = AutoConfig.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=cfg, torch_dtype=torch_dtype)

    meta = {}
    if resume_dir:
        meta_path = os.path.join(resume_dir, "meta.json")
        meta = load_json_if_exists(meta_path) or {}

        model_bin = os.path.join(resume_dir, "pytorch_model.bin")
        if os.path.exists(model_bin):
            sd = torch.load(model_bin, map_location="cpu")
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if is_rank0():
                print(f"[resume] loaded {model_bin}")
                if missing:
                    print(f"[resume] missing keys (showing up to 10): {missing[:10]}")
                if unexpected:
                    print(f"[resume] unexpected keys (showing up to 10): {unexpected[:10]}")
        else:
            if is_rank0():
                print(f"[resume] WARNING: {model_bin} not found; using base weights.")

    return model, tok, meta


def save_step_checkpoint(
    model_fsdp: FSDP,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    out_step_dir: str,
    meta: dict,
    save_optimizer: bool = True,
    save_scheduler: bool = True,
):
    safe_makedirs(out_step_dir)

    # Gather full state dict on rank0
    full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model_fsdp, StateDictType.FULL_STATE_DICT, full_cfg):
        full_sd = model_fsdp.state_dict()

    if is_rank0():
        torch.save(full_sd, os.path.join(out_step_dir, "pytorch_model.bin"))
        if save_optimizer:
            torch.save(optimizer.state_dict(), os.path.join(out_step_dir, "optimizer.pt"))
        if save_scheduler and scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(out_step_dir, "scheduler.pt"))
        with open(os.path.join(out_step_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

        # also save tokenizer files for convenience
        try:
            tokenizer.save_pretrained(out_step_dir)
        except Exception:
            pass


# -------------------------
# Main RMU training
# -------------------------
def build_argparser():
    p = argparse.ArgumentParser()

    # IO
    p.add_argument("--data_path", type=str, required=True, help="Pickle path: list of dicts with prompt/generation/label.")
    p.add_argument("--model_name_or_path", type=str, default="gpt2")
    p.add_argument("--resume_dir", type=str, default=None, help="Step dir containing pytorch_model.bin/optimizer.pt/scheduler.pt/meta.json")
    p.add_argument("--output_dir", type=str, required=True)

    # Data / loader
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)

    # Train
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # RMU
    p.add_argument("--k_schedule", type=str, default="0.75,1.0,1.0,1.0")
    p.add_argument("--alpha", type=float, default=1000.0)
    p.add_argument("--alpha_start", type=float, default=-1.0)
    p.add_argument("--alpha_end", type=float, default=-1.0)
    p.add_argument("--c", type=float, default=4.0)
    p.add_argument("--auto_scale_c", action="store_true")
    p.add_argument("--c_scale", type=float, default=3.0)

    # Optional GA on forget
    p.add_argument("--ga_forget_weight", type=float, default=0.0, help="If >0, add (-w * LM_loss_forget) to total loss (gradient ascent).")

    # FSDP / precision
    p.add_argument("--bf16", action="store_true", help="Use bf16 mixed precision (recommended on A5000).")
    p.add_argument("--seed", type=int, default=42)

    # Logging / saving
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_every_steps", type=int, default=500)
    p.add_argument("--save_optimizer", action="store_true")
    p.add_argument("--save_scheduler", action="store_true")

    # wandb
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)

    return p


def main():
    args = build_argparser().parse_args()

    # local rank (torchrun)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ddp_setup(local_rank)

    set_all_seeds(args.seed + local_rank)

    device = torch.device("cuda", local_rank)
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32

    # --- wandb (rank0 only) ---
    wandb = None
    if args.wandb_project and is_rank0():
        import wandb as _wandb  # local import
        wandb = _wandb
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    # --- load data ---
    import pickle
    with open(args.data_path, "rb") as f:
        data_list = pickle.load(f)
    ds = PromptGenDataset(data_list)

    # split indices
    forget_idx = [i for i, ex in enumerate(data_list) if int(ex["label"]) == 1]
    retain_idx = [i for i, ex in enumerate(data_list) if int(ex["label"]) == 0]
    if is_rank0():
        print(f"[data] total={len(ds)} forget={len(forget_idx)} retain={len(retain_idx)}")

    forget_ds = Subset(ds, forget_idx)
    retain_ds = Subset(ds, retain_idx)

    # --- model/tokenizer/load checkpoint ---
    model, tokenizer, meta = load_model_with_optional_step_checkpoint(
        args.model_name_or_path,
        args.resume_dir,
        torch_dtype=torch_dtype,
    )
    model.to(device)

    # frozen teacher (per-rank full copy; simplest & robust)
    model_frozen = copy.deepcopy(model).to(device)
    model_frozen.eval()
    for p in model_frozen.parameters():
        p.requires_grad_(False)

    # Candidate RMU layers & u-vectors
    layer_names = get_gpt2_rmu_layers(model)
    hidden_size = model.config.n_embd
    u_vectors = create_random_u_vectors(layer_names, hidden_size=hidden_size, device=device)

    # Collator + loaders
    collate = CausalCollator(tokenizer=tokenizer, seq_len=args.seq_len)

    forget_sampler = DistributedSampler(forget_ds, shuffle=True, drop_last=True)
    retain_sampler = DistributedSampler(retain_ds, shuffle=True, drop_last=True)

    forget_loader = DataLoader(
        forget_ds,
        batch_size=args.batch_size,
        sampler=forget_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    retain_loader = DataLoader(
        retain_ds,
        batch_size=args.batch_size,
        sampler=retain_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )

    # --- wrap with FSDP ---
    # GPT-2 block type for auto-wrap
    from transformers.models.gpt2.modeling_gpt2 import GPT2Block

    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        reduce_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        buffer_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )
    auto_wrap = transformer_auto_wrap_policy({GPT2Block})

    model_fsdp = FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device,
        use_orig_params=True,
    )

    # optimizer/scheduler
    optimizer = torch.optim.AdamW(model_fsdp.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # resume optimizer/scheduler if present
    global_step = int(meta.get("global_step", 0)) if isinstance(meta, dict) else 0
    if args.resume_dir:
        opt_sd = torch_load_if_exists(os.path.join(args.resume_dir, "optimizer.pt"), map_location="cpu")
        if opt_sd is not None:
            try:
                optimizer.load_state_dict(opt_sd)
                if is_rank0():
                    print("[resume] optimizer loaded")
            except Exception as e:
                if is_rank0():
                    print(f"[resume] optimizer load failed (continuing): {e}")

    # total steps ~ epochs * max(len_f, len_r) / grad_accum
    steps_per_epoch = max(len(forget_loader), len(retain_loader))
    total_optim_steps = math.ceil(args.epochs * steps_per_epoch / max(1, args.grad_accum))
    warmup_steps = int(args.warmup_ratio * total_optim_steps)

    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_optim_steps)

    if args.resume_dir:
        sch_sd = torch_load_if_exists(os.path.join(args.resume_dir, "scheduler.pt"), map_location="cpu")
        if sch_sd is not None:
            try:
                scheduler.load_state_dict(sch_sd)
                if is_rank0():
                    print("[resume] scheduler loaded")
            except Exception as e:
                if is_rank0():
                    print(f"[resume] scheduler load failed (continuing): {e}")

    # Activation buffers (per step)
    acts: Dict[str, torch.Tensor] = {}

    def hook_factory(key: str):
        def _hook(_m, _inp, out):
            acts[key] = out
        return _hook

    k_schedule = parse_k_schedule(args.k_schedule, args.epochs)

    model_fsdp.train()

    # main training
    for epoch in range(args.epochs):
        forget_sampler.set_epoch(epoch)
        retain_sampler.set_epoch(epoch)

        k = k_schedule[epoch]
        start_i, end_i = get_layer_range_indices(k, len(layer_names))
        sampled_i = random.randint(start_i, end_i)
        sampled_layer = layer_names[sampled_i]

        alpha_e = linear_alpha(epoch, args.epochs, args.alpha, args.alpha_start, args.alpha_end)

        if args.auto_scale_c:
            rms = compute_layer_rms_gpt2(model_frozen, sampled_layer, retain_loader, device=device, num_batches=2)
            c_e = args.c_scale * max(rms, 1e-6)
        else:
            c_e = args.c

        # register hooks on the sampled layer
        h_updated = model_fsdp.get_submodule(sampled_layer).register_forward_hook(hook_factory("updated"))
        h_frozen = model_frozen.get_submodule(sampled_layer).register_forward_hook(hook_factory("frozen"))

        main_iter, num_steps = pair_full_epoch(forget_loader, retain_loader)

        if is_rank0():
            print(
                f"[epoch {epoch+1}/{args.epochs}] layer={sampled_layer} "
                f"range=[{start_i},{end_i}] steps={num_steps} alpha={alpha_e:.2f} c={c_e:.4f} k={k}"
            )

        pbar = tqdm(total=num_steps, disable=not is_rank0(), leave=False)

        running = {
            "loss_total": 0.0,
            "loss_forget_rmu": 0.0,
            "loss_retain_match": 0.0,
            "loss_forget_lm": 0.0,
        }
        denom = 0

        for step_i, (fb, rb) in enumerate(main_iter, start=1):
            # move batches
            fb = {k: v.to(device, non_blocking=True) for k, v in fb.items() if isinstance(v, torch.Tensor)}
            rb = {k: v.to(device, non_blocking=True) for k, v in rb.items() if isinstance(v, torch.Tensor)}

            # grad accumulation: use no_sync for micro-steps
            accum_idx = (step_i - 1) % args.grad_accum
            do_step = (accum_idx == args.grad_accum - 1) or (step_i == num_steps)

            ctx = model_fsdp.no_sync() if (args.grad_accum > 1 and not do_step) else torch.enable_grad()
            with ctx:
                # clear activation dict each micro-step
                acts.clear()

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if args.bf16 else torch.float32):
                    # FORGET pass (updated model)
                    out_f = model_fsdp(
                        input_ids=fb["input_ids"],
                        attention_mask=fb["attention_mask"],
                        labels=fb["labels"],
                    )
                    act_f = acts.get("updated", None)
                    if act_f is None:
                        continue

                    # RMU forget loss: push activations toward c*u
                    u = u_vectors[sampled_layer]  # (1,1,C)
                    # act_f is typically (B,T,C). Broadcast u over (B,T,*)
                    loss_forget_rmu = torch.mean((act_f - (c_e * u)) ** 2)

                    # Optional: GA on forget LM loss (gradient ascent => subtract)
                    loss_forget_lm = out_f.loss
                    ga_term = -args.ga_forget_weight * loss_forget_lm if args.ga_forget_weight > 0 else 0.0

                    # RETAIN pass (frozen + updated)
                    _ = model_frozen(input_ids=rb["input_ids"], attention_mask=rb["attention_mask"])
                    act_r_frozen = acts.get("frozen", None)
                    _ = model_fsdp(input_ids=rb["input_ids"], attention_mask=rb["attention_mask"])
                    act_r_updated = acts.get("updated", None)
                    if act_r_frozen is None or act_r_updated is None:
                        continue

                    loss_retain = torch.mean((act_r_updated - act_r_frozen) ** 2)

                    loss = loss_forget_rmu + alpha_e * loss_retain + ga_term
                    loss = loss / max(1, args.grad_accum)

                loss.backward()

            if do_step:
                if args.grad_clip and args.grad_clip > 0:
                    model_fsdp.clip_grad_norm_(args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            # logging (reduce to mean across ranks)
            with torch.no_grad():
                lt = torch.tensor(float(loss_forget_rmu.detach().float().item()), device=device)
                lr = torch.tensor(float(loss_retain.detach().float().item()), device=device)
                lf = torch.tensor(float(loss_forget_lm.detach().float().item()), device=device)
                ltot = torch.tensor(float((loss.detach().float().item()) * max(1, args.grad_accum)), device=device)

                lt = all_reduce_mean(lt).item()
                lr = all_reduce_mean(lr).item()
                lf = all_reduce_mean(lf).item()
                ltot = all_reduce_mean(ltot).item()

                running["loss_total"] += ltot
                running["loss_forget_rmu"] += lt
                running["loss_retain_match"] += lr
                running["loss_forget_lm"] += lf
                denom += 1

                if is_rank0():
                    pbar.update(1)

                if wandb is not None and is_rank0() and (global_step % args.log_every == 0) and do_step:
                    wandb.log(
                        {
                            "train/loss_total": ltot,
                            "train/loss_forget_rmu": lt,
                            "train/loss_retain_match": lr,
                            "train/loss_forget_lm": lf,
                            "train/alpha": alpha_e,
                            "train/c": c_e,
                            "train/k": k,
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/epoch": epoch + 1,
                            "train/global_step": global_step,
                            "train/sampled_layer": sampled_i,
                        },
                        step=global_step,
                    )

            # save
            if do_step and (global_step % args.save_every_steps == 0):
                meta_out = {
                    "global_step": global_step,
                    "epoch": epoch + 1,
                    "sampled_layer": sampled_layer,
                    "alpha": alpha_e,
                    "c": c_e,
                    "k": k,
                    "time": time.time(),
                }
                step_dir = os.path.join(args.output_dir, f"step_{global_step:08d}")
                save_step_checkpoint(
                    model_fsdp,
                    tokenizer,
                    optimizer,
                    scheduler,
                    step_dir,
                    meta_out,
                    save_optimizer=args.save_optimizer,
                    save_scheduler=args.save_scheduler,
                )
                if is_rank0():
                    print(f"[save] {step_dir}")

        if is_rank0():
            pbar.close()
            if denom > 0:
                print(
                    f"[epoch {epoch+1}] "
                    f"avg_total={running['loss_total']/denom:.4f} "
                    f"avg_forget_rmu={running['loss_forget_rmu']/denom:.4f} "
                    f"avg_retain={running['loss_retain_match']/denom:.6f} "
                    f"avg_forget_lm={running['loss_forget_lm']/denom:.4f}"
                )

        # remove hooks
        h_updated.remove()
        h_frozen.remove()

    # final save
    final_meta = {"global_step": global_step, "time": time.time(), "done": True}
    final_dir = os.path.join(args.output_dir, f"final_step_{global_step:08d}")
    save_step_checkpoint(
        model_fsdp,
        tokenizer,
        optimizer,
        scheduler,
        final_dir,
        final_meta,
        save_optimizer=args.save_optimizer,
        save_scheduler=args.save_scheduler,
    )
    if is_rank0():
        print(f"[final] saved to {final_dir}")
        if wandb is not None:
            wandb.finish()

    ddp_cleanup()


if __name__ == "__main__":
    main()
