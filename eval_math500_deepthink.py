#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# -----------------------------
# Your format tokens
# -----------------------------
REASONING_START = "<start_deepthink>"
REASONING_END   = "<end_deepthink>"
SOLUTION_START  = "<SOLUTION>"
SOLUTION_END    = "</SOLUTION>"

DEFAULT_SYSTEM_PROMPT = f"""User asks a question, and the Assistant solves it.

Format requirements:
1. Enclose the reasoning process within {REASONING_START} ... {REASONING_END}.
2. Enclose the final answer within {SOLUTION_START} ... {SOLUTION_END}.

Constraints on the Answer:
- For numerical answers, use digits (e.g., 42).
- For fractions or algebraic expressions, use standard LaTeX format (e.g., \\frac{{1}}{{2}}, 2x+5).
- Output NOTHING else inside the solution tags except the value itself.
"""


# -----------------------------
# Answer parsing / normalization
# -----------------------------
BOXED_ANY_RE = re.compile(r"\\boxed\{(.+?)\}")
FRAC_RE = re.compile(r"\\frac\{([-+]?\d+)\}\{([-+]?\d+)\}")
PLAIN_FRAC_RE = re.compile(r"^([-+]?\d+)\/([-+]?\d+)$")
FLOAT_RE = re.compile(r"^[-+]?\d+(\.\d+)?$")

def extract_pred_answer(text: str) -> Optional[str]:
    # 1) Prefer <SOLUTION> ... </SOLUTION> (allow end tag missing due to stop removal)
    if SOLUTION_START in text:
        seg = text.split(SOLUTION_START, 1)[1]
        if SOLUTION_END in seg:
            seg = seg.split(SOLUTION_END, 1)[0]
        return seg.strip()

    # 2) \boxed{...}
    m = BOXED_ANY_RE.search(text)
    if m:
        return m.group(1).strip()

    # 3) fallback: None
    return None

def is_format_ok(text: str) -> bool:
    # same spirit as your GSM8K script: require at least reasoning + solution start
    return (REASONING_START in text) and (SOLUTION_START in text)

def _strip_math_delims(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == "$" and s[-1] == "$":
        s = s[1:-1].strip()
    return s

def normalize_answer(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = _strip_math_delims(s)

    # unwrap boxed if the whole string is boxed
    m = re.fullmatch(r"\\boxed\{(.+)\}", s.strip())
    if m:
        s = m.group(1).strip()

    # remove \left \right
    s = s.replace("\\left", "").replace("\\right", "")

    # remove common latex spacing
    s = s.replace("\\,", "").replace("\\!", "")

    # collapse whitespace
    s = re.sub(r"\s+", "", s)
    return s

def try_parse_number(norm: Optional[str]) -> Optional[float]:
    if norm is None:
        return None

    # \frac{a}{b}
    m = FRAC_RE.fullmatch(norm)
    if m:
        a = int(m.group(1))
        b = int(m.group(2))
        if b != 0:
            return a / b

    # a/b
    m = PLAIN_FRAC_RE.fullmatch(norm)
    if m:
        a = int(m.group(1))
        b = int(m.group(2))
        if b != 0:
            return a / b

    # integer/float
    if FLOAT_RE.fullmatch(norm):
        return float(norm)

    return None

def answers_match(gold_raw: str, pred_raw: Optional[str], numeric_tol: float = 1e-9) -> bool:
    g = normalize_answer(gold_raw)
    p = normalize_answer(pred_raw)

    if g is None or p is None:
        return False

    # exact normalized match
    if g == p:
        return True

    # numeric fallback (helps when gold is "0.5" and pred is "1/2", etc.)
    gn = try_parse_number(g)
    pn = try_parse_number(p)
    if gn is not None and pn is not None:
        return abs(gn - pn) <= numeric_tol

    return False


# -----------------------------
# Prompt building
# -----------------------------
def build_messages(question: str, system_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

def build_plain_prompt(question: str, system_prompt: str) -> str:
    return f"{system_prompt}\n\nUser: {question}\nAssistant:"

def build_prompts(tokenizer, questions: List[str], system_prompt: str, force_prefix: bool) -> List[str]:
    """
    force_prefix=True: append '<start_deepthink>\n' right after generation prompt.
    This improves compliance with your RL format (same trick as your GSM8K script).
    """
    if hasattr(tokenizer, "apply_chat_template"):
        msgs = [build_messages(q, system_prompt) for q in questions]
        rendered = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        if force_prefix:
            rendered = [r + "\n" + REASONING_START + "\n" for r in rendered]
        return rendered
    else:
        prompts = [build_plain_prompt(q, system_prompt) for q in questions]
        if force_prefix:
            prompts = [p + "\n" + REASONING_START + "\n" for p in prompts]
        return prompts


# -----------------------------
# vLLM runner
# -----------------------------
@dataclass
class GenConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0

def map_dtype_to_vllm(dtype: str) -> str:
    if dtype == "fp16":
        return "half"
    if dtype == "bf16":
        return "bfloat16"
    if dtype == "fp32":
        return "float32"
    return "bfloat16"

def make_vllm_engine(
    model_path: str,
    vllm_dtype: str,
    gpu_memory_utilization: float = 0.90,
    tensor_parallel_size: int = 1,
    max_model_len: Optional[int] = None,
) -> LLM:
    kwargs = dict(
        model=model_path,
        dtype=vllm_dtype,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
    )
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    return LLM(**kwargs)

def generate_batch_vllm(
    llm: LLM,
    tokenizer,
    questions: List[str],
    system_prompt: str,
    gen_cfg: GenConfig,
    force_prefix: bool,
) -> List[str]:
    prompts = build_prompts(tokenizer, questions, system_prompt, force_prefix=force_prefix)

    sp = SamplingParams(
        max_tokens=gen_cfg.max_new_tokens,
        temperature=gen_cfg.temperature,
        top_p=gen_cfg.top_p,
        # stop 会被 vLLM 截断（通常不包含 stop 字符串本身），所以解析时不要强依赖 </SOLUTION>
        stop=[SOLUTION_END, "<|endoftext|>"],
    )

    outs = llm.generate(prompts, sp)

    # Keep order stable
    try:
        outs = sorted(outs, key=lambda o: int(o.request_id))
    except Exception:
        pass

    return [o.outputs[0].text for o in outs]


# -----------------------------
# Evaluation (single model)
# -----------------------------
def evaluate_math500_vllm(
    model_path: str,
    examples: List[Dict[str, Any]],
    system_prompt: str,
    batch_size: int,
    gen_cfg: GenConfig,
    hf_tokenizer_name_or_path: str,
    vllm_dtype: str,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    max_model_len: Optional[int],
    force_prefix: bool,
    strict_format: bool,
) -> Tuple[float, float, List[Dict[str, Any]]]:

    tok = AutoTokenizer.from_pretrained(hf_tokenizer_name_or_path, use_fast=True, trust_remote_code=True)

    llm = make_vllm_engine(
        model_path=model_path,
        vllm_dtype=vllm_dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
    )

    results: List[Dict[str, Any]] = []
    correct = 0
    fmt_ok_cnt = 0
    total = len(examples)

    pbar = tqdm(total=total, desc=f"Evaluating {model_path}", unit="q", dynamic_ncols=True)

    for start in range(0, total, batch_size):
        batch = examples[start:start + batch_size]
        qs = [x["problem"] for x in batch]
        golds = [x["answer"] for x in batch]

        pred_texts = generate_batch_vllm(
            llm=llm,
            tokenizer=tok,
            questions=qs,
            system_prompt=system_prompt,
            gen_cfg=gen_cfg,
            force_prefix=force_prefix,
        )

        for ex, gold, pred_text in zip(batch, golds, pred_texts):
            fmt_ok = is_format_ok(pred_text)
            fmt_ok_cnt += int(fmt_ok)

            pred_ans = extract_pred_answer(pred_text) if (fmt_ok or not strict_format) else None
            ok = answers_match(gold, pred_ans)

            correct += int(ok)

            results.append({
                "unique_id": ex.get("unique_id"),
                "subject": ex.get("subject"),
                "level": ex.get("level"),
                "problem": ex.get("problem"),
                "gold": gold,
                "pred": pred_ans,
                "pred_text": pred_text,
                "correct": ok,
                "format_ok": fmt_ok,
            })

            done = len(results)
            pbar.set_postfix(
                acc=f"{(correct / done):.4f}",
                fmt=f"{(fmt_ok_cnt / done):.3f}",
            )
            pbar.update(1)

    pbar.close()

    acc = correct / total if total else 0.0
    fmt_rate = fmt_ok_cnt / total if total else 0.0
    return acc, fmt_rate, results


def main():
    p = argparse.ArgumentParser()

    # model(s)
    p.add_argument("--model", type=str, required=True, help="Model path (merged model is recommended for vLLM)")
    p.add_argument("--compare_model", type=str, default="", help="Optional second model to compare")

    # dataset
    p.add_argument("--dataset_id", type=str, default="HuggingFaceH4/MATH-500")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--limit", type=int, default=-1)

    # decoding
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)

    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)

    # vLLM engine params
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--max_model_len", type=int, default=0, help="0 means let vLLM decide")

    # format controls
    p.add_argument("--force_prefix", action="store_true", help="Append <start_deepthink> before generation.")
    p.add_argument("--strict_format", action="store_true", help="If no <SOLUTION>, count as wrong.")

    # outputs
    p.add_argument("--save_jsonl", type=str, default="", help="Save per-example outputs to jsonl")
    p.add_argument("--save_jsonl_compare", type=str, default="", help="Save compare outputs to jsonl (if compare_model set)")
    args = p.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    ds = load_dataset(args.dataset_id)
    if args.split not in ds:
        raise ValueError(f"Split '{args.split}' not found in dataset '{args.dataset_id}'. Available: {list(ds.keys())}")
    data = ds[args.split]
    if args.limit and args.limit > 0:
        data = data.select(range(min(args.limit, len(data))))

    # pull all fields needed
    examples = [{
        "problem": x["problem"],
        "answer": x["answer"],
        "subject": x.get("subject"),
        "level": x.get("level"),
        "unique_id": x.get("unique_id"),
    } for x in data]

    gen_cfg = GenConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    vllm_dtype = map_dtype_to_vllm(args.dtype)
    max_model_len = args.max_model_len if args.max_model_len and args.max_model_len > 0 else None

    print(f"Loaded {args.dataset_id}/{args.split}: {len(examples)} examples")
    print(f"Decoding: max_new_tokens={args.max_new_tokens}, temperature={args.temperature}, top_p={args.top_p}")
    print(f"vLLM: dtype={vllm_dtype}, gpu_memory_utilization={args.gpu_memory_utilization}, "
          f"tensor_parallel_size={args.tensor_parallel_size}, max_model_len={max_model_len}")
    print(f"Format: force_prefix={args.force_prefix}, strict_format={args.strict_format}")
    print("-" * 80)

    acc, fmt_rate, res = evaluate_math500_vllm(
        model_path=args.model,
        examples=examples,
        system_prompt=args.system_prompt,
        batch_size=args.batch_size,
        gen_cfg=gen_cfg,
        hf_tokenizer_name_or_path=args.model,
        vllm_dtype=vllm_dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=max_model_len,
        force_prefix=args.force_prefix,
        strict_format=args.strict_format,
    )
    print(f"[MODEL] {args.model}\n  Accuracy (normalized exact / numeric-fallback): {acc:.4f}\n  Format OK rate: {fmt_rate:.3f}")
    print("-" * 80)

    if args.save_jsonl:
        os.makedirs(os.path.dirname(args.save_jsonl) or ".", exist_ok=True)
        with open(args.save_jsonl, "w", encoding="utf-8") as f:
            for i, row in enumerate(res):
                row_out = dict(idx=i, **row)
                f.write(json.dumps(row_out, ensure_ascii=False) + "\n")
        print(f"Saved: {args.save_jsonl}")
        print("-" * 80)

    if args.compare_model:
        acc2, fmt2, res2 = evaluate_math500_vllm(
            model_path=args.compare_model,
            examples=examples,
            system_prompt=args.system_prompt,
            batch_size=args.batch_size,
            gen_cfg=gen_cfg,
            hf_tokenizer_name_or_path=args.compare_model,
            vllm_dtype=vllm_dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=max_model_len,
            force_prefix=args.force_prefix,
            strict_format=args.strict_format,
        )
        print(f"[COMPARE] {args.compare_model}\n  Accuracy: {acc2:.4f}\n  Format OK rate: {fmt2:.3f}")
        print("-" * 80)
        print(f"Delta (MODEL - COMPARE): {acc - acc2:+.4f}")

        if args.save_jsonl_compare:
            os.makedirs(os.path.dirname(args.save_jsonl_compare) or ".", exist_ok=True)
            with open(args.save_jsonl_compare, "w", encoding="utf-8") as f:
                for i, (a, b) in enumerate(zip(res, res2)):
                    out = {
                        "idx": i,
                        "unique_id": a.get("unique_id"),
                        "subject": a.get("subject"),
                        "level": a.get("level"),
                        "problem": a.get("problem"),
                        "gold": a.get("gold"),
                        "model_pred": a.get("pred"),
                        "model_correct": a.get("correct"),
                        "model_format_ok": a.get("format_ok"),
                        "model_pred_text": a.get("pred_text"),
                        "compare_pred": b.get("pred"),
                        "compare_correct": b.get("correct"),
                        "compare_format_ok": b.get("format_ok"),
                        "compare_pred_text": b.get("pred_text"),
                    }
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"Saved compare jsonl: {args.save_jsonl_compare}")

if __name__ == "__main__":
    main()
