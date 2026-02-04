#!/usr/bin/env python3
"""
Toxic-only perplexity on google/civil_comments.

- Filters examples by toxicity >= --tox_threshold
- Concatenates tokenized toxic texts (EOS between samples)
- Computes NLL/PPL using a sliding window (seq_len, stride)
- Supports:
  * HF model name/path
  * raw checkpoint dirs with pytorch_model.bin (needs --base_model)

Example:
  python toxic_ppl_civil_comments.py \
    --model ./ckpts/train_lt_256/step_00000484 \
    --base_model gpt2 \
    --split test \
    --tox_threshold 0.5 \
    --max_toxic_samples 5000 \
    --seq_len 1024 --stride 512
"""

import argparse
import itertools
import json
import math
import os
from typing import List, Optional, Tuple

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="HF model name/path OR raw ckpt dir with pytorch_model.bin")
    p.add_argument("--base_model", type=str, default=None, help="Required if --model is a raw state_dict dir (no config.json)")
    p.add_argument("--split", type=str, default="test", help="civil_comments split (often train/validation/test)")
    p.add_argument("--streaming", action="store_true", help="Use streaming mode (recommended if disk is tight)")

    p.add_argument("--text_field", type=str, default="text")
    p.add_argument("--tox_field", type=str, default="toxicity")
    p.add_argument("--tox_threshold", type=float, default=0.5)

    p.add_argument("--max_toxic_samples", type=int, default=5000, help="How many toxic examples to use")
    p.add_argument("--max_total_scanned", type=int, default=200000, help="Cap on how many rows to scan to find toxic samples")
    p.add_argument("--max_tokens", type=int, default=0, help="Optional cap on total tokens after concat (0 = no cap)")

    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--stride", type=int, default=512)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    p.add_argument("--json", action="store_true", help="Print a single JSON line (nice for bash loops)")

    # tqdm controls
    p.add_argument("--tqdm", action="store_true", help="Enable tqdm progress bars")
    p.add_argument("--tqdm_update", type=int, default=50, help="Update tqdm postfix every N steps (avoid overhead)")

    return p.parse_args()


def pick_dtype(dtype_str: str) -> Optional[torch.dtype]:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(dtype_str, None)


def is_hf_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json"))


def load_model_and_tokenizer(model_arg: str, base_model: Optional[str], device: str, torch_dtype: Optional[torch.dtype]):
    # Case 1: HF hub name OR a directory with config.json
    if (not os.path.isdir(model_arg)) or is_hf_dir(model_arg):
        tok = AutoTokenizer.from_pretrained(model_arg, use_fast=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained(model_arg, torch_dtype=torch_dtype).to(device).eval()
        return mdl, tok

    # Case 2: raw checkpoint directory containing pytorch_model.bin only
    state_path = os.path.join(model_arg, "pytorch_model.bin")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Expected {state_path} in checkpoint dir.")

    if base_model is None:
        raise ValueError("Checkpoint dir has no config.json. Provide --base_model (e.g., gpt2).")

    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype).to(device)
    sd = torch.load(state_path, map_location="cpu")
    missing, unexpected = mdl.load_state_dict(sd, strict=False)
    mdl.eval()

    if len(missing) or len(unexpected):
        print(f"[warn] load_state_dict(strict=False): missing={len(missing)} unexpected={len(unexpected)}")

    return mdl, tok


def collect_toxic_texts(
    ds,
    text_field: str,
    tox_field: str,
    tox_threshold: float,
    max_toxic_samples: int,
    max_total_scanned: int,
    use_tqdm: bool = True,
    tqdm_update: int = 50,
) -> Tuple[List[str], int, float]:
    toxic_texts: List[str] = []
    scanned = 0
    tox_sum = 0.0

    # IMPORTANT: cap progress total to max_total_scanned (not full dataset length)
    # so tqdm never displays huge totals like 319,895 when you only want to scan up to N.
    scan_cap = max_total_scanned

    iterator = itertools.islice(iter(ds), max_total_scanned)

    pbar = None
    if use_tqdm:
        # If streaming, total can still be shown as scan_cap since we are hard-capping the scan.
        pbar = tqdm(total=scan_cap, desc="Scanning for toxic samples", unit="ex")

    for ex in iterator:
        scanned += 1

        tox = ex.get(tox_field, None)
        txt = ex.get(text_field, None)

        if isinstance(txt, str) and txt.strip() and tox is not None:
            try:
                tox_val = float(tox)
            except Exception:
                tox_val = None

            if tox_val is not None and tox_val >= tox_threshold:
                toxic_texts.append(txt)
                tox_sum += tox_val

        if pbar:
            pbar.update(1)
            if scanned % max(1, tqdm_update) == 0:
                pbar.set_postfix(
                    scanned=scanned,
                    toxic=len(toxic_texts),
                    avg_tox=(tox_sum / len(toxic_texts)) if toxic_texts else 0.0,
                )

        if len(toxic_texts) >= max_toxic_samples:
            break

    if pbar:
        # If we stopped early (hit max_toxic_samples), move bar to completion for a clean UI.
        if scanned < scan_cap:
            pbar.update(scan_cap - scanned)
        pbar.close()

    avg_tox = (tox_sum / len(toxic_texts)) if toxic_texts else 0.0
    return toxic_texts, scanned, avg_tox


def main():
    args = parse_args()
    torch.set_grad_enabled(False)

    torch_dtype = pick_dtype(args.dtype)
    model, tokenizer = load_model_and_tokenizer(args.model, args.base_model, args.device, torch_dtype)

    # Load dataset
    ds = load_dataset("google/civil_comments", split=args.split, streaming=args.streaming)

    toxic_texts, scanned, avg_tox = collect_toxic_texts(
        ds,
        text_field=args.text_field,
        tox_field=args.tox_field,
        tox_threshold=args.tox_threshold,
        max_toxic_samples=args.max_toxic_samples,
        max_total_scanned=args.max_total_scanned,
        use_tqdm=args.tqdm,
        tqdm_update=args.tqdm_update,
    )

    if not toxic_texts:
        raise RuntimeError(
            f"No toxic examples found with {args.tox_field} >= {args.tox_threshold} "
            f"after scanning {scanned} rows. Try lowering --tox_threshold or raising --max_total_scanned."
        )

    eos_id = tokenizer.eos_token_id
    all_ids: List[int] = []

    tok_pbar = tqdm(total=len(toxic_texts), desc="Tokenizing toxic texts", unit="txt") if args.tqdm else None
    for i, t in enumerate(toxic_texts, start=1):
        ids = tokenizer(t, add_special_tokens=False).input_ids
        if ids:
            all_ids.extend(ids)
            if eos_id is not None:
                all_ids.append(eos_id)

        if tok_pbar:
            tok_pbar.update(1)
            if i % max(1, args.tqdm_update) == 0:
                tok_pbar.set_postfix(tokens=len(all_ids))

        if args.max_tokens and len(all_ids) >= args.max_tokens:
            all_ids = all_ids[: args.max_tokens]
            break

    if tok_pbar:
        tok_pbar.close()

    enc = torch.tensor(all_ids, dtype=torch.long)
    if enc.numel() < 2:
        raise RuntimeError("Not enough tokens to compute perplexity after filtering.")

    seq_len, stride = args.seq_len, args.stride
    if stride <= 0 or stride > seq_len:
        raise ValueError("--stride must be in (0, seq_len].")

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0

    total_windows = (enc.size(0) + stride - 1) // stride
    win_pbar = tqdm(total=total_windows, desc="Scoring windows", unit="win") if args.tqdm else None

    for w_i, begin_loc in enumerate(range(0, enc.size(0), stride), start=1):
        end_loc = min(begin_loc + seq_len, enc.size(0))
        input_ids = enc[begin_loc:end_loc].unsqueeze(0).to(args.device)

        trg_len = end_loc - prev_end_loc
        labels = input_ids.clone()
        if trg_len < labels.size(1):
            labels[:, :-trg_len] = -100

        out = model(input_ids=input_ids, labels=labels)

        nll_sum += out.loss.item() * trg_len
        n_tokens += trg_len
        prev_end_loc = end_loc

        if win_pbar:
            win_pbar.update(1)
            if w_i % max(1, args.tqdm_update) == 0:
                win_pbar.set_postfix(tokens_scored=n_tokens, avg_nll=(nll_sum / max(1, n_tokens)))

        if end_loc == enc.size(0):
            break

    if win_pbar:
        win_pbar.close()

    avg_nll = nll_sum / max(1, n_tokens)
    ppl = math.exp(avg_nll)

    result = {
        "model": args.model,
        "base_model": args.base_model,
        "dataset": "google/civil_comments",
        "split": args.split,
        "tox_field": args.tox_field,
        "tox_threshold": args.tox_threshold,
        "toxic_examples_used": len(toxic_texts),
        "rows_scanned": scanned,
        "avg_toxicity_of_used": avg_tox,
        "tokens_scored": n_tokens,
        "avg_nll": avg_nll,
        "perplexity": ppl,
    }

    if args.json:
        print(json.dumps(result))
    else:
        print(f"Dataset: google/civil_comments split={args.split}")
        print(f"Toxic filter: {args.tox_field} >= {args.tox_threshold}")
        print(f"Rows scanned: {scanned} | Toxic examples used: {len(toxic_texts)} | Avg toxicity (used): {avg_tox:.4f}")
        print(f"Tokens scored: {n_tokens}")
        print(f"Avg NLL/token: {avg_nll:.6f}")
        print(f"Perplexity: {ppl:.4f}")


if __name__ == "__main__":
    main()
