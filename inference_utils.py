# inference_toxicity.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

def _is_hf_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json"))

def _load_generator(
    model_name_or_path: str,
    base_model_name_if_state_dict: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Union[str, Dict[str, int]] = "auto",
):
    """
    Loads a causal LM generator from:
      - a HF model name (e.g., "gpt2")
      - a HF checkpoint dir (has config.json)
      - a raw state_dict file or dir (e.g., pytorch_model.bin) *if* base_model_name_if_state_dict is provided
    """
    torch_dtype = torch_dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)

    # Try standard HF load first
    try:
        tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        gen = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        gen.eval()
        return tok, gen
    except Exception:
        pass

    # If not loadable directly, try state_dict route
    if base_model_name_if_state_dict is None:
        raise ValueError(
            f"Could not load '{model_name_or_path}' via from_pretrained. "
            f"If this is a raw state_dict checkpoint, pass base_model_name_if_state_dict="
            f"(e.g., 'gpt2' / 'gpt2-medium' / etc.)."
        )

    base_tok = AutoTokenizer.from_pretrained(base_model_name_if_state_dict, use_fast=True)
    if base_tok.pad_token_id is None:
        base_tok.pad_token = base_tok.eos_token

    base_cfg = AutoConfig.from_pretrained(base_model_name_if_state_dict)
    gen = AutoModelForCausalLM.from_config(base_cfg)

    # find state_dict
    sd_path = model_name_or_path
    if os.path.isdir(sd_path):
        # common filenames
        for cand in ["pytorch_model.bin", "model.pt", "model.bin"]:
            p = os.path.join(sd_path, cand)
            if os.path.exists(p):
                sd_path = p
                break

    state = torch.load(sd_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    missing, unexpected = gen.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] Missing keys while loading state_dict (showing first 20): {missing[:20]}")
    if unexpected:
        print(f"[warn] Unexpected keys while loading state_dict (showing first 20): {unexpected[:20]}")

    gen.to("cuda" if torch.cuda.is_available() else "cpu")
    gen = gen.to(dtype=torch_dtype)
    gen.eval()
    return base_tok, gen


def _load_toxicity_scorer(
    toxicity_model_name: str = "unitary/unbiased-toxic-roberta",
    torch_dtype: Optional[torch.dtype] = None,
):
    """
    Loads a local toxicity classifier. Default is a widely used HF toxicity model.
    Returns (tokenizer, model, toxic_label_idx_or_none, label_names).
    """
    torch_dtype = torch_dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)
    tok = AutoTokenizer.from_pretrained(toxicity_model_name, use_fast=True)
    clf = AutoModelForSequenceClassification.from_pretrained(
        toxicity_model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    clf.eval()

    # Try to locate a single "toxicity"/"toxic" label index if present
    id2label = getattr(clf.config, "id2label", None) or {}
    label_names = [id2label.get(i, str(i)) for i in range(clf.config.num_labels)]
    low = [n.lower() for n in label_names]
    tox_idx = None
    for key in ["toxicity", "toxic"]:
        if key in low:
            tox_idx = low.index(key)
            break

    return tok, clf, tox_idx, label_names


@torch.inference_mode()
def _toxicity_scores(
    texts: List[str],
    tox_tok,
    tox_model,
    tox_idx: Optional[int],
    batch_size: int = 32,
    max_length: int = 512,
) -> List[float]:
    """
    Returns one toxicity score per text.
    If the classifier has a specific toxicity/toxic label, uses that sigmoid prob.
    Otherwise returns max sigmoid prob across labels as a fallback.
    """
    scores: List[float] = []
    device = next(tox_model.parameters()).device

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tox_tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        logits = tox_model(**enc).logits  # (B, C)
        probs = torch.sigmoid(logits)

        if tox_idx is not None and probs.shape[1] > tox_idx:
            s = probs[:, tox_idx]
        else:
            s = probs.max(dim=1).values

        scores.extend(s.detach().float().cpu().tolist())

    return scores
from typing import List, Optional, Dict, Any
import torch
from detoxify import Detoxify

def load_detoxify(model_name: str = "unbiased", device: Optional[str] = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return Detoxify(model_name, device=device)

def detoxify_toxicity_scores(
    texts: List[str],
    detox_model,
    batch_size: int = 32,
    score_key: str = "auto",  # "auto" or explicit key like "toxicity"/"toxic"
) -> List[float]:
    """
    Returns a single toxicity score per text.
    - 'unbiased' / 'multilingual' typically provide 'toxicity'
    - 'original' typically provides 'toxic'
    (Detoxify returns different label sets by challenge/model.) :contentReference[oaicite:2]{index=2}
    """
    out_scores: List[float] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        preds: Dict[str, Any] = detox_model.predict(batch)  # dict(label -> list[float])

        if score_key != "auto":
            if score_key not in preds:
                raise KeyError(f"Requested score_key='{score_key}' not in Detoxify outputs: {list(preds.keys())}")
            out_scores.extend([float(x) for x in preds[score_key]])
            continue

        # auto: prefer "toxicity" then "toxic", else fallback to max over all labels
        if "toxicity" in preds:
            out_scores.extend([float(x) for x in preds["toxicity"]])
        elif "toxic" in preds:
            out_scores.extend([float(x) for x in preds["toxic"]])
        else:
            # max over labels for each example
            keys = list(preds.keys())
            # preds[k] is list[float] len=batch
            for j in range(len(batch)):
                out_scores.append(float(max(preds[k][j] for k in keys)))

    return out_scores



@torch.inference_mode()
def infer_and_score_toxicity(
    model_name_or_path: str,
    test_prompts: List[Dict[str, Any]],
    *,
    prompt_key: str = "prompt",
    base_model_name_if_state_dict: Optional[str] = None,
    # generation
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.95,
    do_sample: bool = True,
    repetition_penalty: Optional[float] = None,
    # toxicity scoring
    toxicity_model_name: str = "unitary/unbiased-toxic-roberta",
    toxicity_batch_size: int = 32,
    score_on: str = "completion",  # "completion" or "full"
    # runtime
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Union[str, Dict[str, int]] = "auto",
) -> List[Dict[str, Any]]:
    """
    Args:
      model_name_or_path: HF name ("gpt2") or HF checkpoint dir, or raw state_dict path/dir (then pass base_model_name_if_state_dict).
      test_prompts: list of dicts, each containing prompt_key.
      score_on: "completion" scores toxicity on generated continuation only; "full" scores on prompt+completion.

    Returns:
      list of dicts with: prompt, generated_text, completion_text, toxicity_score
    """
    assert score_on in {"completion", "full"}

    gen_tok, gen_model = _load_generator(
        model_name_or_path,
        base_model_name_if_state_dict=base_model_name_if_state_dict,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    tox_tok, tox_model, tox_idx, tox_labels = _load_toxicity_scorer(
        toxicity_model_name=toxicity_model_name,
        torch_dtype=torch_dtype,
    )

    results: List[Dict[str, Any]] = []
    texts_for_scoring: List[str] = []

    pbar = tqdm(test_prompts, desc="Generating", unit="ex")
    for item in pbar:
        prompt = item.get(prompt_key, "")
        if not isinstance(prompt, str):
            prompt = str(prompt)

        enc = gen_tok(prompt, return_tensors="pt")
        # Move to generator device
        gen_device = next(gen_model.parameters()).device
        enc = {k: v.to(gen_device) for k, v in enc.items()}

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=gen_tok.pad_token_id,
            eos_token_id=gen_tok.eos_token_id,
        )
        # remove Nones
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        if repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = repetition_penalty

        out = gen_model.generate(**enc, **gen_kwargs)

        full_text = gen_tok.decode(out[0], skip_special_tokens=True)

        # Attempt to isolate completion by slicing tokens
        prompt_len = enc["input_ids"].shape[1]
        completion_ids = out[0][prompt_len:]
        completion_text = gen_tok.decode(completion_ids, skip_special_tokens=True)

        to_score = completion_text if score_on == "completion" else full_text
        texts_for_scoring.append(to_score)

        results.append(
            {
                "prompt": prompt,
                "generated_text": full_text,
                "completion_text": completion_text,
                # filled after scoring
                "toxicity_score": None,
                "toxicity_model": toxicity_model_name,
                "toxicity_label_used": (tox_labels[tox_idx] if tox_idx is not None else "max_over_labels"),
                "score_on": score_on,
            }
        )

    # Toxicity scoring (batched)
    tox_scores = _toxicity_scores(
        texts_for_scoring,
        tox_tok=tox_tok,
        tox_model=tox_model,
        tox_idx=tox_idx,
        batch_size=toxicity_batch_size,
    )

    for r, s in zip(results, tox_scores):
        r["toxicity_score"] = float(s)

    return results
