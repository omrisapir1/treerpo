
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from treerpo.eval.gsm8k import evaluate_gsm8k


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on GSM8K")

    parser.add_argument("--model_id", type=str, required=True,
                       help="HF model id or local path")
    parser.add_argument("--tokenizer_id", type=str, default=None,
                       help="Tokenizer id (defaults to model_id)")

    parser.add_argument("--gpu_memory_utilization", type=float, default=0.4,
                       help="vLLM GPU memory utilization [0,1]")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                       help="vLLM dtype: auto|bfloat16|float16|float32")
    parser.add_argument("--max_model_len", type=int, default=None,
                       help="Override vLLM max model length")
    parser.add_argument("--max_num_seqs", type=int, default=256,
                       help="vLLM max concurrent sequences")
    parser.add_argument("--max_num_batched_tokens", type=int, default=None,
                       help="vLLM max batched tokens")
    parser.add_argument("--trust_remote_code", action="store_true",
                       help="Trust remote code when loading model")

    parser.add_argument("--max_samples", type=int, default=None,
                       help="Limit test samples for faster evaluation")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for inference")

    parser.add_argument("--greedy_max_new_tokens", type=int, default=1024,
                       help="Max new tokens for greedy decoding")

    # Majority voting
    parser.add_argument("--majk", type=int, default=8,
                       help="K for majority voting (Maj@K)")
    parser.add_argument("--majk_temperature", type=float, default=0.7,
                       help="Temperature for Maj@K sampling")
    parser.add_argument("--majk_top_p", type=float, default=0.8,
                       help="Top-p for Maj@K sampling")
    parser.add_argument("--majk_max_new_tokens", type=int, default=1024,
                       help="Max new tokens for Maj@K decoding")

    args = parser.parse_args()

    llm_kwargs = {
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
    }

    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens

    print("=" * 60)
    print("GSM8K EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model_id}")
    print(f"Tokenizer: {args.tokenizer_id or args.model_id}")
    print(f"Max samples: {args.max_samples or 'all (1319)'}")
    print(f"Batch size: {args.batch_size}")
    print(f"vLLM dtype: {args.dtype}")
    print(f"GPU memory util: {args.gpu_memory_utilization}")
    print("-" * 60)
    print(f"Greedy: max_tokens={args.greedy_max_new_tokens}")
    print(f"Maj@{args.majk}: temp={args.majk_temperature}, top_p={args.majk_top_p}, max_tokens={args.majk_max_new_tokens}")
    print("-" * 60)

    try:
        results = evaluate_gsm8k(
            model_id=args.model_id,
            tokenizer_id=args.tokenizer_id,
            greedy_max_new_tokens=args.greedy_max_new_tokens,
            majk=args.majk,
            majk_temperature=args.majk_temperature,
            majk_top_p=args.majk_top_p,
            majk_max_new_tokens=args.majk_max_new_tokens,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            llm_kwargs=llm_kwargs,
        )

        print("=" * 60)
        print("FINAL RESULTS:")
        print("=" * 60)
        print(f"Greedy Accuracy:  {results['greedy_acc']:.4f} ({results['greedy_acc']*100:.2f}%)")
        print(f"Maj@{args.majk} Accuracy: {results['majk_acc']:.4f} ({results['majk_acc']*100:.2f}%)")
        print(f"Total samples:    {results['n']}")
        print("=" * 60)

        return results

    except Exception as e:
        print(f"ERROR: Evaluation failed with: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
