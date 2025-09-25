
from __future__ import annotations
import random
from collections import Counter
from typing import Optional, List, Dict

from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from treerpo.utils.prompts import build_prompt_ids
from treerpo.utils.answers import extract_final_answer, math_equal


def _parse_ref_from_solution(solution: str) -> str:
    """
    GSM8K 'answer' field ends with a line like: '#### 42'
    We take whatever appears after the last '####'.
    """
    if solution is None:
        return ""
    marker = "####"
    idx = solution.rfind(marker)
    if idx == -1:
        return solution.strip()
    return solution[idx + len(marker):].strip()


def _make_prompts(tokenizer, questions: List[str]) -> List[str]:
    """
    Build chat-style prompts using the shared helper, then decode to raw text
    for vLLM input.
    """
    prompts = []
    for q in questions:
        ids = build_prompt_ids(tokenizer, q)  # List[int]
        text = tokenizer.decode(ids)
        prompts.append(text)
    return prompts


def _majority_vote(strings: List[str]) -> Optional[str]:
    """
    Majority vote over strings. If tie, pick randomly among tied items.
    Returns None if the list is empty.
    """
    if not strings:
        return None
    counts = Counter(strings)
    top_count = max(counts.values())
    tied = [s for s, c in counts.items() if c == top_count]
    return random.choice(tied)


def evaluate_gsm8k(
    model_id: str,
    tokenizer_id: Optional[str] = None,
    *,
    greedy_max_new_tokens: int = 1024,
    majk: int = 8,
    majk_temperature: float = 0.7,
    majk_top_p: float = 0.8,
    majk_max_new_tokens: int = 1024,
    batch_size: int = 64,
    max_samples: Optional[int] = None,
    llm_kwargs: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    Evaluate a model on GSM8K test set (Greedy and Maj@K) using vLLM.

    Args:
        model_id: vLLM model ID/path.
        tokenizer_id: tokenizer to use for chat template. If None, defaults to model_id.
        greedy_max_new_tokens: max_new_tokens for greedy decoding (temp=0.0).
        majk: K for Maj@K.
        majk_temperature: temperature for Maj@K sampling.
        majk_top_p: top-p for Maj@K sampling.
        majk_max_new_tokens: max_new_tokens for Maj@K decoding.
        batch_size: number of prompts per vLLM.generate call.
        llm_kwargs: extra kwargs forwarded to vLLM LLM(...), e.g., {"dtype": "bfloat16"}.

    Returns:
        {"greedy_acc": float, "majk_acc": float, "n": int}
    """
    if tokenizer_id is None:
        tokenizer_id = model_id

    # Init tokenizer (trust_remote_code=True is often needed for chat templates)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)

    # Load GSM8K test split
    ds = load_dataset("openai/gsm8k", "main", split="test")

    if max_samples is not None:
        max_samples = min(max_samples, len(ds))
        ds = ds.select(range(max_samples))
    questions = [ex["question"] for ex in ds]
    refs = [_parse_ref_from_solution(ex["answer"]) for ex in ds]
    n_total = len(questions)

    # Build prompts once
    prompts = _make_prompts(tokenizer, questions)

    # Init vLLM
    llm_kwargs = llm_kwargs or {}
    llm = LLM(model=model_id, **llm_kwargs)

    # ---------------- Greedy (temp=0.0) ----------------
    greedy_params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=greedy_max_new_tokens,
    )

    greedy_correct = 0
    for i in tqdm(range(0, n_total, batch_size), desc="Greedy"):
        batch_prompts = prompts[i : i + batch_size]
        outputs = llm.generate(batch_prompts, greedy_params)
        # outputs is a list of RequestOutput; each has .outputs[0].text
        preds_text = [out.outputs[0].text for out in outputs]
        # Extract boxed and score
        for j, pred in enumerate(preds_text):
            boxed = extract_final_answer(pred)
            # Rule: no boxed → incorrect
            correct = bool(boxed and math_equal(boxed, refs[i + j]))
            greedy_correct += int(correct)

    greedy_acc = greedy_correct / n_total

    # ---------------- Maj@K ----------------
    majk_params = SamplingParams(
        n=majk,
        temperature=majk_temperature,
        top_p=majk_top_p,
        max_tokens=majk_max_new_tokens,
    )

    majk_correct = 0
    for i in tqdm(range(0, n_total, batch_size), desc=f"Maj@{majk}"):
        batch_prompts = prompts[i : i + batch_size]
        outputs = llm.generate(batch_prompts, majk_params)
        # For each prompt, we have K outputs in out.outputs
        for b, out in enumerate(outputs):
            candidates = [cand.text for cand in out.outputs]
            boxed_preds = [extract_final_answer(t) for t in candidates]
            # Drop None (no boxed) early
            boxed_preds = [bp for bp in boxed_preds if bp is not None]
            if not boxed_preds:
                # No boxed at all -> incorrect
                continue
            vote = _majority_vote(boxed_preds)  # tie breaks randomly
            if vote is not None and math_equal(vote, refs[i + b]):
                majk_correct += 1

    majk_acc = majk_correct / n_total

    print(f"\nGSM8K (test) | Greedy acc: {greedy_acc:.4f} | Maj@{majk} acc: {majk_acc:.4f} | N={n_total}\n")
    return {"greedy_acc": greedy_acc, "majk_acc": majk_acc, "n": n_total}
