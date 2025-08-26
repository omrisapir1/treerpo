
import argparse
import sys
from typing import Optional

import torch

from .config import TreeRPOConfig
from .trainers.treerpo_trainer import TreeRPOTrainer
from .eval.gsm8k import evaluate_gsm8k


# ------------------------------- helpers ---------------------------------
def _dtype_from_str(s: str) -> torch.dtype:
    s = s.lower()
    if s in ("auto",):
        # let the trainer/model handle "auto" path
        return "auto"  # type: ignore[return-value]
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if s not in mapping:
        raise ValueError(f"Unsupported --torch_dtype '{s}'. Use one of: auto,bfloat16,float16,float32")
    return mapping[s]


# ------------------------------ subcommands ------------------------------
def add_train_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "train",
        help="Train with TreeRPOTrainer (single-GPU, no DeepSpeed).",
    )

    # --- model / tokenizer ---
    p.add_argument("--model_id", type=str, required=True, help="HF model id or local path.")
    p.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        help="Load dtype for model weights: auto|bfloat16|float16|float32 (default: bfloat16).",
    )

    # --- data ---
    p.add_argument("--dataset", type=str, required=True, help="HF dataset id for training (expects 'question' & 'final_answer').")
    p.add_argument("--max_train_samples", type=int, default=None, help="Optional cap on training samples for quick runs.")

    # --- trainer core args (only the ones TreeRPOTrainer handles) ---
    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--lr_scheduler_type", type=str, default="linear")

    # --- logging / saving ---
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--logging_steps", type=int, default=50)

    # --- TreeRPO specifics ---
    p.add_argument("--beta", type=float, default=0.0, help="KL coefficient (0 disables ref model).")
    p.add_argument("--epsilon", type=float, default=0.2, help="Clipping epsilon.")

    # vLLM engine (minimal knobs)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.4, help="vLLM GPU mem utilization [0,1].")
    p.add_argument("--enable_prefix_caching", action="store_true", help="Enable vLLM prefix caching (recommended).")

    # tree parameters (defaults per your spec)
    p.add_argument("--max_depth", type=int, default=7, help="k: maximum branching depth (depth < k can split).")
    p.add_argument("--min_segment_len", type=int, default=150, help="L_min: tokens before a node is eligible to split.")
    p.add_argument("--entropy_threshold", type=float, default=1.0, help="H_th: entropy gate at split point.")
    p.add_argument("--entropy_topk", type=int, default=20, help="Top-K for entropy computation.")
    p.add_argument("--coverage_min_chars", type=int, default=150, help="S_min for coverage split.")
    p.add_argument("--coverage_children_max", type=int, default=2, help="Max coverage children.")

    # decoding params for node continuations
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.85)
    p.add_argument("--top_k", type=int, default=25)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--max_new_tokens", type=int, default=1300)

    p.set_defaults(func=cmd_train)


def add_eval_gsm8k_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "eval-gsm8k",
        help="Evaluate on openai/gsm8k test split (vLLM-only). Prints greedy_acc and maj@8_acc.",
    )

    # model / engine
    p.add_argument("--model_id", type=str, required=True, help="HF model id or local path.")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.4, help="vLLM GPU mem utilization [0,1].")
    p.add_argument("--dtype", type=str, default="bfloat16", help="vLLM dtype: auto|bfloat16|float16|float32.")
    p.add_argument("--max_model_len", type=int, default=None, help="Optional vLLM max model length override.")

    # dataset slice
    p.add_argument("--max_samples", type=int, default=None, help="Limit number of test examples for quicker runs.")

    p.set_defaults(func=cmd_eval_gsm8k)


# --------------------------------- handlers ---------------------------------
def cmd_train(args: argparse.Namespace) -> None:
    # Build TreeRPOConfig
    # Always trust_remote_code=True; pass model_init_kwargs with dtype
    model_init_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": _dtype_from_str(args.torch_dtype),
        "use_cache": True,  # trainer will disable if gradient checkpointing was ever enabled (not exposed here)
    }

    treerpo_args = TreeRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,

        # generation path (we use vLLM inside the trainer)
        use_vllm=True,  # TreeRPO uses vLLM for tree building
        model_init_kwargs=model_init_kwargs,

        # KL / clipping
        beta=args.beta,
        epsilon=args.epsilon,

        # Tree parameters
        treerpo_max_depth=args.max_depth,
        treerpo_min_segment_len=args.min_segment_len,
        treerpo_entropy_threshold=args.entropy_threshold,
        treerpo_entropy_topk=args.entropy_topk,
        treerpo_coverage_min_chars=args.coverage_min_chars,
        treerpo_coverage_children_max=args.coverage_children_max,

        # Node decoding params
        treerpo_temperature=args.temperature,
        treerpo_top_p=args.top_p,
        treerpo_top_k=args.top_k,
        treerpo_repetition_penalty=args.repetition_penalty,
        treerpo_max_new_tokens=args.max_new_tokens,

        # vLLM minimal knobs
        vllm_gpu_memory_utilization=args.gpu_memory_utilization,
        vllm_enable_prefix_caching=args.enable_prefix_caching,
    )

    # Load dataset (the trainer expects samples with 'question' and 'final_answer')
    # We only support HF datasets string here.
    from datasets import load_dataset

    ds = load_dataset(args.dataset)
    if "train" in ds:
        train_ds = ds["train"]
    else:
        # fallback: a single split dataset
        first_split = list(ds.keys())[0]
        train_ds = ds[first_split]

    if args.max_train_samples is not None:
        train_ds = train_ds.select(range(min(len(train_ds), args.max_train_samples)))

    # Minimal sanity checks on columns (no remapping by design)
    for col in ("question", "final_answer"):
        if col not in train_ds.column_names:
            raise ValueError(
                f"Dataset '{args.dataset}' must contain column '{col}'. "
                "No column remapping is supported by this CLI."
            )

    # Instantiate trainer and run
    trainer = TreeRPOTrainer(
        model=args.model_id,
        args=treerpo_args,
        train_dataset=train_ds,
        eval_dataset=None,
        callbacks=None,
        optimizers=(None, None),
    )

    trainer.train()
    trainer.save_model(args.output_dir)  # final save (weights + tokenizer via trainer)


def cmd_eval_gsm8k(args: argparse.Namespace) -> None:
    # vLLM-only evaluation; trust_remote_code is handled inside evaluate_gsm8k
    greedy_acc, majk_acc = evaluate_gsm8k(
        model_id=args.model_id,
    )
    print(f"greedy_acc: {greedy_acc:.4f}")
    print(f"maj@8_acc: {majk_acc:.4f}")


# ----------------------------------- main -----------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="treerpo",
        description="TreeRPO: Hierarchical credit assignment for reasoning LMs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_train_subparser(subparsers)
    add_eval_gsm8k_subparser(subparsers)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
        return 0
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    except Exception as e:
        # keep it simple / readable
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
