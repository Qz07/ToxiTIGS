#!/usr/bin/env python3
"""
Perplexity over a pickle dataset (list[dict]).

- Loads a pickle file containing list[dict]
- (Optional) filters examples by tox_field >= tox_threshold
- Uses text_field as corpus
- Concatenates tokenized texts (EOS between samples)
- Computes NLL/PPL using a sliding window (seq_len, stride)

Example:
  python ppl_from_pickle.py \
    --data_pickle /path/to/data.pkl \
    --model ./ckpts/train_lt_256/step_00000484 \
    --base_model gpt2 \
    --tox_threshold 0.5 \
    --max_samples 5000 \
    --seq_len 1024 --stride 512 \
    --tqdm
"""

import argparse
import json
import math
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--data_pickle", type=str, required=True, help="Path to pickle file containing list[dict]")
    p.add_argument("--text_field", type=str, default="text", help="Key in each dict for the text")
    p.add_argument("--tox_field", type=str, default="toxicity", help="Key in each dict for toxicity score")
    p.add_argument("--tox_threshold", type=float, default=0.5, help="Keep examples with tox >= threshold")
    p.add_argument(
        "--disable_tox_filter",
        action="store_true",
        help="If set, ignore tox_field/tox_threshold and use ALL examples with non-empty text.",
    )

    p.add_argument("--max_samples", type=int, default=5000, help="Max number of examples to use (after filtering)")
    p.add_argument(
        "--max_total_scanned",
        type=int,
        default=0,
        help="Cap how many rows to scan (0 = scan all). Useful if pickle is huge.",
    )
    p.add_argument("--max_tokens", type=int, default=0, help="Optional cap on total tokens after concat (0 = no cap)")

    # model
    p.add_argument("--model", type=str, required=True, help="HF model name/path OR raw ckpt dir with pytorch_model.bin")
    p.add_argument("--base_model", type=str, default=None, help="Required if --model is a raw state_dict dir")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])

    # ppl windowing
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--stride", type=int, default=512)

    # output / progress
    p.add_argument("--json", action="store_true", help="Print a single JSON line")
    p.add_argument("--tqdm", action="store_true", help="Enable tqdm progress bars")
    p.add_argument("--tqdm_update", type=int, default=50, help="Update tqdm postfix every N steps")

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


def load_pickle_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "rb") as f:
        obj = pickle.load(f)

    # common cases
    if isinstance(obj, list):
        if all(isinstance(x, dict) for x in obj):
            return obj
        raise TypeError("Pickle contained a list, but not a list of dicts.")

    if isinstance(obj, dict):
        for k in ("data", "records", "examples", "items"):
            v = obj.get(k, None)
            if isinstance(v, list) and all(isinstance(x, dict) for x in v):
                return v
        raise TypeError("Pickle contained a dict, but no list[dict] found under keys: data/records/examples/items.")

    # optional: pandas DataFrame support (if pickle stored a DF)
    try:
        import pandas as pd  # type: ignore
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
    except Exception:
        pass

    raise TypeError(f"Unsupported pickle object type: {type(obj)}. Expected list[dict] (or DataFrame).")


def collect_texts_from_records(
    records: List[Dict[str, Any]],
    text_field: str,
    tox_field: str,
    tox_threshold: float,
    disable_tox_filter: bool,
    max_samples: int,
    max_total_scanned: int,
    use_tqdm: bool,
    tqdm_update: int,
) -> Tuple[List[str], int, float]:
    texts: List[str] = []
    scanned = 0
    tox_sum = 0.0

    scan_cap = len(records) if max_total_scanned in (0, None) else min(len(records), max_total_scanned)

    pbar = tqdm(total=scan_cap, desc="Scanning records", unit="ex") if use_tqdm else None

    for ex in records[:scan_cap]:
        scanned += 1
        txt = ex.get(text_field, None)

        if not (isinstance(txt, str) and txt.strip()):
            if pbar:
                pbar.update(1)
            continue

        if disable_tox_filter:
            texts.append(txt)
        else:
            tox = ex.get(tox_field, None)
            tox_val = None
            if tox is not None:
                try:
                    tox_val = float(tox)
                except Exception:
                    tox_val = None

            if tox_val is not None and tox_val >= tox_threshold:
                texts.append(txt)
                tox_sum += tox_val

        if pbar:
            pbar.update(1)
            if scanned % max(1, tqdm_update) == 0:
                pbar.set_postfix(
                    scanned=scanned,
                    kept=len(texts),
                    avg_tox=(tox_sum / len(texts)) if (texts and not disable_tox_filter) else 0.0,
                )

        if len(texts) >= max_samples:
            break

    if pbar:
        # finish bar for cleaner UI
        if scanned < scan_cap:
            pbar.update(scan_cap - scanned)
        pbar.close()

    avg_tox = (tox_sum / len(texts)) if (texts and not disable_tox_filter) else 0.0
    return texts, scanned, avg_tox


def main():
    args = parse_args()
    torch.set_grad_enabled(False)

    torch_dtype = pick_dtype(args.dtype)
    model, tokenizer = load_model_and_tokenizer(args.model, args.base_model, args.device, torch_dtype)

    records = load_pickle_records(args.data_pickle)

    texts, scanned, avg_tox = collect_texts_from_records(
        records=records,
        text_field=args.text_field,
        tox_field=args.tox_field,
        tox_threshold=args.tox_threshold,
        disable_tox_filter=args.disable_tox_filter,
        max_samples=args.max_samples,
        max_total_scanned=args.max_total_scanned,
        use_tqdm=args.tqdm,
        tqdm_update=args.tqdm_update,
    )

    if not texts:
        if args.disable_tox_filter:
            raise RuntimeError(f"No non-empty '{args.text_field}' strings found after scanning {scanned} rows.")
        raise RuntimeError(
            f"No examples found with {args.tox_field} >= {args.tox_threshold} "
            f"after scanning {scanned} rows. Try lowering --tox_threshold, using --disable_tox_filter, "
            f"or increasing --max_total_scanned."
        )

    eos_id = tokenizer.eos_token_id
    all_ids: List[int] = []

    tok_pbar = tqdm(total=len(texts), desc="Tokenizing texts", unit="txt") if args.tqdm else None
    for i, t in enumerate(texts, start=1):
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
        raise RuntimeError("Not enough tokens to compute perplexity after filtering/concatenation.")

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
        "data_pickle": args.data_pickle,
        "text_field": args.text_field,
        "tox_field": args.tox_field,
        "tox_threshold": None if args.disable_tox_filter else args.tox_threshold,
        "examples_used": len(texts),
        "rows_scanned": scanned,
        "avg_toxicity_of_used": avg_tox,
        "tokens_scored": n_tokens,
        "avg_nll": avg_nll,
        "perplexity": ppl,
    }

    if args.json:
        print(json.dumps(result))
    else:
        print(f"Data: {args.data_pickle}")
        print(f"Examples scanned: {scanned} | Examples used: {len(texts)}")
        if not args.disable_tox_filter:
            print(f"Filter: {args.tox_field} >= {args.tox_threshold} | Avg toxicity (used): {avg_tox:.4f}")
        print(f"Tokens scored: {n_tokens}")
        print(f"Avg NLL/token: {avg_nll:.6f}")
        print(f"Perplexity: {ppl:.4f}")


if __name__ == "__main__":
    main()
