from dataclasses import dataclass, field
from typing import Optional, Any, Dict, Tuple

from transformers import TrainingArguments


@dataclass
class TreeRPOConfig(TrainingArguments):
    r"""
    Configuration class for the TreeRPO trainer.

    Only TreeRPO-specific parameters are listed here. For the rest, see
    `transformers.TrainingArguments`.

    Notes:
    - Single-GPU by design (no DeepSpeed).
    - KL is disabled by default (`beta=0.0`).
    - vLLM is used locally (AsyncLLMEngine), not via an external server.
    """

    # ---------------- Model loading ----------------
    _n_gpu: int = field(init=True, repr=False, default=1)
    model_init_kwargs: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Keyword arguments passed to `AutoModelForCausalLM.from_pretrained` when the model is a string."
        },
    )

    # ---------------- Dataset / preprocessing ----------------
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={"help": "Keep extra dataset columns (e.g., `question`, `final_answer`)."},
    )

    # ---------------- Generation (policy sampling) ----------------
    temperature: float = field(
        default=0.6,
        metadata={"help": "Sampling temperature used by the policy during tree expansion."},
    )
    top_p: float = field(
        default=0.85,
        metadata={"help": "Top-p nucleus sampling for policy sampling during tree expansion."},
    )
    top_k: Optional[int] = field(
        default=25,
        metadata={"help": "Top-k sampling for policy sampling during tree expansion."},
    )
    min_p: Optional[float] = field(
        default=None,
        metadata={"help": "Min-p sampling (optional). Leave None to disable."},
    )
    repetition_penalty: float = field(
        default=1.1,
        metadata={"help": "Repetition penalty for generation during tree expansion."},
    )
    max_completion_length: int = field(
        default=1300,
        metadata={"help": "Per-node max new tokens during generation (tree expansion)."},
    )
    max_full_answer_length: int = field(
        default=2600,
        metadata={"help": "Max total tokens during generation per answer."},
    )
    cache_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "HF cache implementation if vLLM is disabled."},
    )

    # ---------------- Tree policy (split rules) ----------------
    split_delimiters: Tuple[str, ...] = field(
        default=("\n", "!", ".", "?", ";", ":"),
        metadata={"help": "Characters considered as split boundaries for delimiter-aware branching."},
    )
    coverage_children_max: int = field(
        default=4,
        metadata={"help": "Maximum number of post-hoc coverage children to spawn."},
    )
    max_depth: int = field(
        default=7,
        metadata={"help": "Maximum tree depth (k)."},
    )
    min_segment_len: int = field(
        default=150,
        metadata={"help": "Minimum tokens in a segment before entropy-based split (L_min)."},
    )
    entropy_top_k: int = field(
        default=20,
        metadata={"help": "Top-k logits used to compute next-token entropy."},
    )
    entropy_threshold: float = field(
        default=1.0,
        metadata={"help": "Entropy threshold (H_th) to trigger a split at a delimiter."},
    )
    coverage_min_chars: int = field(
        default=150,
        metadata={
            "help": "Coverage split: if the final leaf exceeds this char length (S_min), "
                    "rewind to parent and add extra sibling(s)."
        },
    )
    max_async_streams: int = field(
        default=128,
        metadata={"help": "Max concurrent async decode streams for the tree builder."},
    )
    forced_root_split: bool = field(
        default=True,
        metadata={"help": "Force exactly two children at the root before any token is generated."},
    )

    # ---------------- vLLM local engine ----------------
    use_vllm: bool = field(
        default=True,
        metadata={"help": "Use local AsyncLLMEngine (vLLM) for fast tree expansion."},
    )
    vllm_gpu_mem_util: float = field(
        default=0.4,
        metadata={"help": "vLLM GPU memory utilization (0..1)."},
    )
    vllm_max_num_seqs: int = field(
        default=128,
        metadata={"help": "vLLM `max_num_seqs`."},
    )
    vllm_max_batched_tokens: int = field(
        default=8 * 3000,
        metadata={"help": "vLLM `max_num_batched_tokens`."},
    )
    vllm_enable_prefix_caching: bool = field(
        default = True,
        metadata = {"help": "Enable vLLM prefix caching."},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Pass `trust_remote_code` to model/tokenizer loaders and vLLM."},
    )

    # ---------------- RL objective ----------------
    beta: float = field(
        default=0.0,
        metadata={"help": "KL coefficient. 0 disables KL and the reference model."},
    )
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Clipping epsilon (lower bound)."},
    )
    epsilon_high: Optional[float] = field(
        default=None,
        metadata={"help": "Upper clipping epsilon. If None, equals `epsilon`."},
    )

    # ---------------- Compute safeguards ----------------
    max_elements_per_forward: int = field(
        default=1200,
        metadata={
            "help": "If B*L exceeds this threshold, compute per-row to reduce memory (avoid OOM)."
        },
    )

    # ---------------- Minimal logging ----------------
    log_tree_stats: bool = field(
        default=True,
        metadata={"help": "Log simple tree stats (e.g., avg leaves per prompt)."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.epsilon_high is None:
            self.epsilon_high = self.epsilon

        # Basic sanity checks
        if not (0.0 < self.temperature):
            raise ValueError("temperature must be > 0.")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1].")
        if self.top_k is not None and self.top_k < 0:
            raise ValueError("top_k must be None or >= 0.")
        if self.min_p is not None and not (0.0 <= self.min_p <= 1.0):
            raise ValueError("min_p must be in [0, 1].")
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1.")
        if self.min_segment_len < 0 or self.coverage_min_chars < 0:
            raise ValueError("min_segment_len and coverage_min_chars must be >= 0.")
        if not (0.0 < self.vllm_gpu_mem_util <= 1.0):
            raise ValueError("vLLM GPU memory utilization must be in (0, 1].")
