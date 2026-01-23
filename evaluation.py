#!/usr/bin/env python3
"""
evaluate_toxicity.py

Usage examples:
  python evaluate_toxicity.py \
    --model gpt2 \
    --data /path/to/test_prompts.pkl

  # If your model path is a raw state_dict (not HF directory), provide base model:
  python evaluate_toxicity.py \
    --model /path/to/step_00010000/pytorch_model.bin \
    --base_model gpt2 \
    --data /path/to/test_prompts.pkl

Notes:
- Expects pickle file: List[Dict], each dict has key "text" (the prompt).
- Uses infer_and_score_toxicity from inference_toxicity.py
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Import from your script (make sure inference_toxicity.py is in the same folder or on PYTHONPATH)
from inference_utils import infer_and_score_toxicity


def load_pickle_list_of_dicts(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, list):
        raise TypeError(f"Expected pickle to contain a list, got {type(obj)}")

    out: List[Dict[str, Any]] = []
    for i, item in enumerate(obj):
        if not isinstance(item, dict):
            raise TypeError(f"Item {i} is not a dict (got {type(item)}).")
        if "text" not in item:
            raise KeyError(f"Item {i} missing required key 'text'. Keys={list(item.keys())}")
        out.append(item)
    return out


def summarize(scores: List[float]) -> Dict[str, float]:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }

    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="HF model name/path OR checkpoint dir OR state_dict path")
    parser.add_argument("--data", type=str, required=True, help="Path to pickle file: List[Dict] with key 'text'")
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="If --model is a raw state_dict (not HF), set base model name (e.g., gpt2, gpt2-medium).",
    )

    # Generation params (defaults match your inference)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling (default: False).")
    parser.add_argument("--no_do_sample", action="store_true", help="Force greedy decoding.")
    parser.add_argument("--repetition_penalty", type=float, default=None)

    # Toxicity scoring
    parser.add_argument("--toxicity_model", type=str, default="unitary/unbiased-toxic-roberta")
    parser.add_argument("--toxicity_batch_size", type=int, default=32)
    parser.add_argument("--score_on", type=str, default="completion", choices=["completion", "full"])

    # Runtime
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype. 'auto' picks float16 on CUDA else float32.",
    )
    args = parser.parse_args()

    # Resolve do_sample flag
    if args.no_do_sample:
        do_sample = False
    elif args.do_sample:
        do_sample = True
    else:
        # default in your inference is True; keep that behavior
        do_sample = True

    # dtype resolution
    if args.dtype == "auto":
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device_str} dtype={torch_dtype} score_on={args.score_on}")
    print(f"[info] model={args.model}")
    print(f"[info] data={args.data}")

    test_prompts = load_pickle_list_of_dicts(args.data)

    # Run inference + toxicity scoring (generation happens per-example in your function; scoring is batched)
    results = infer_and_score_toxicity(
        model_name_or_path=args.model,
        test_prompts=test_prompts,
        prompt_key="text",
        base_model_name_if_state_dict=args.base_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=do_sample,
        repetition_penalty=args.repetition_penalty,
        toxicity_model_name=args.toxicity_model,
        toxicity_batch_size=args.toxicity_batch_size,
        score_on=args.score_on,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    scores = [float(r["toxicity_score"]) for r in results if r.get("toxicity_score") is not None]
    stats = summarize(scores)

    # Nicely formatted output
    print("\n" + "=" * 72)
    print("TOXICITY EVALUATION SUMMARY")
    print("=" * 72)
    print(f"Examples (n)            : {stats['n']}")
    print(f"Mean toxicity           : {stats['mean']:.6f}")
    print(f"Std toxicity            : {stats['std']:.6f}")
    print(f"Min / Median / Max      : {stats['min']:.6f} / {stats['p50']:.6f} / {stats['max']:.6f}")
    print(f"P90 / P95               : {stats['p90']:.6f} / {stats['p95']:.6f}")

    # Add a short descriptor of what was scored
    if results:
        print(f"Toxicity model          : {results[0].get('toxicity_model')}")
        print(f"Label used              : {results[0].get('toxicity_label_used')}")
        print(f"Scored on               : {results[0].get('score_on')}")

    print("=" * 72)

    # Optional: show a few worst examples
    # (Keeps output readable; comment out if you don't want it.)
    worst_k = min(5, len(results))
    if worst_k > 0:
        worst = sorted(results, key=lambda r: float(r["toxicity_score"]), reverse=True)[:worst_k]
        print("\nTop worst (highest toxicity) examples:")
        for i, r in enumerate(worst, 1):
            p = r["prompt"].replace("\n", " ")[:200]
            c = r["completion_text"].replace("\n", " ")[:200]
            print(f"  {i:02d}. score={float(r['toxicity_score']):.6f}")
            print(f"      prompt     : {p}")
            print(f"      completion : {c}")
            if len(p) >= 200 or len(c) >= 200:
                print("      ...")
        print()

    # Ensure clean shutdown of CUDA context in some environments
    if torch.cuda.is_available():
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
