#!/usr/bin/env python3
"""
Simple training script for TreeRPO.
Usage: python train.py --model_id <model> --dataset <dataset> --output_dir <dir>
"""

import argparse
import torch
from datasets import load_dataset

from treerpo.config import TreeRPOConfig
from treerpo.trainers.treerpo_trainer import TreeRPOTrainer


def _dtype_from_str(s: str) -> torch.dtype:
    """Convert string to torch dtype."""
    s = s.lower()
    if s in ("auto",):
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


def main():
    parser = argparse.ArgumentParser(description="Train TreeRPO model")

    # --- Model/Tokenizer ---
    parser.add_argument("--model_id", type=str, required=True, help="HF model id or local path")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                       help="Model dtype: auto|bfloat16|float16|float32")

    # --- Data ---
    parser.add_argument("--dataset", type=str, required=True,
                       help="HF dataset id (expects 'question' & 'final_answer' columns)")
    parser.add_argument("--max_train_samples", type=int, default=None,
                       help="Limit training samples for quick runs")

    # --- Training Args ---
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")

    # --- Logging/Saving ---
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=50)

    # --- TreeRPO Specifics ---
    parser.add_argument("--beta", type=float, default=0.0, help="KL coefficient")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Clipping epsilon")

    # --- vLLM Engine ---
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.35)
    parser.add_argument("--enable_prefix_caching", action="store_true")

    # --- Tree Parameters ---
    parser.add_argument("--max_depth", type=int, default=7)
    parser.add_argument("--min_segment_len", type=int, default=150)
    parser.add_argument("--entropy_threshold", type=float, default=1.0)
    parser.add_argument("--entropy_top_k", type=int, default=20)
    parser.add_argument("--coverage_min_chars", type=int, default=150)
    parser.add_argument("--coverage_children_max", type=int, default=2)

    # --- Generation Parameters ---
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--top_k", type=int, default=25)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--max_completion_length", type=int, default=1300)

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="train")
    if args.max_train_samples is not None:
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))
    print(f"Training on {len(dataset)} samples")

    # Build TreeRPOConfig
    model_init_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": _dtype_from_str(args.torch_dtype),
        "use_cache": True,
    }

    config = TreeRPOConfig(
        # Basic training args
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

        # vLLM settings
        use_vllm=True,
        model_init_kwargs=model_init_kwargs,
        vllm_gpu_mem_util=args.gpu_memory_utilization,
        trust_remote_code=True,

        # TreeRPO settings
        beta=args.beta,
        epsilon=args.epsilon,

        # Tree parameters
        max_depth=args.max_depth,
        min_segment_len=args.min_segment_len,
        entropy_threshold=args.entropy_threshold,
        entropy_top_k=args.entropy_top_k,
        coverage_min_chars=args.coverage_min_chars,
        coverage_children_max=args.coverage_children_max,

        # Generation parameters
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        max_completion_length=args.max_completion_length,
    )

    # Create trainer
    print(f"Initializing TreeRPO trainer with model: {args.model_id}")
    trainer = TreeRPOTrainer(
        model=args.model_id,
        args=config,
        train_dataset=dataset,
    )

    # Train
    print("Starting training...")
    trainer.train()

    print("Training completed!")


if __name__ == "__main__":
    main()
