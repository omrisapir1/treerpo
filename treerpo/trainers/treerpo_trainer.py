
import os
import asyncio
from collections import defaultdict
from typing import Any, Optional, Union, List, Dict

import torch
from torch import nn
from datasets import Dataset, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

from treerpo import TreeRPOConfig
from treerpo.tree_builder.entropy_tree import TreeBuilder


# ------------------------------ utilities ------------------------------ #

def set_seed_all(seed: int):
    # keep: minimal, device-specific
    from accelerate.utils import set_seed
    set_seed(seed, device_specific=True)


def pad_1d(list_of_1d_tensors: List[torch.Tensor], padding_value: int) -> torch.Tensor:
    """Left-align pad 1D tensors to the same length."""
    if not list_of_1d_tensors:
        return torch.empty(0, dtype=torch.long)
    max_len = max(x.numel() for x in list_of_1d_tensors)
    out = torch.full(
        (len(list_of_1d_tensors), max_len),
        padding_value,
        dtype=list_of_1d_tensors[0].dtype,
        device=list_of_1d_tensors[0].device if list_of_1d_tensors[0].is_cuda else None,
    )
    for i, t in enumerate(list_of_1d_tensors):
        out[i, : t.numel()] = t
    return out


def _to_vllm_dtype_str(torch_dtype: Optional[torch.dtype], model: nn.Module) -> str:
    """Map torch dtype (or infer from model) to vLLM dtype string."""
    MAP = {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
        torch.float64: "float64",
    }
    if torch_dtype in MAP:
        return MAP[torch_dtype]
    try:
        p = next(model.parameters())
        return MAP.get(p.dtype, "auto")
    except StopIteration:
        return "auto"


def _run_coro_blocking(coro):
    """Run an async coroutine from sync code, safely."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        return fut.result()
    else:
        return asyncio.run(coro)


def _find_vllm_client(engine: AsyncLLMEngine):
    """
    Find an object exposing `reset_prefix_cache()` and `collective_rpc()`
    across vLLM versions.
    """
    if hasattr(engine, "reset_prefix_cache") and hasattr(engine, "collective_rpc"):
        return engine
    for attr in ("_engine", "_client", "_engine_client", "_rpc_client"):
        obj = getattr(engine, attr, None)
        if obj and hasattr(obj, "reset_prefix_cache") and hasattr(obj, "collective_rpc"):
            return obj
    return None


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ---------------------------- main trainer ----------------------------- #

class TreeRPOTrainer(Trainer):
    """
    Minimal single-GPU TreeRPO trainer.
    - no DeepSpeed, no W&B
    - KL penalty optional (beta=0 by default)
    - no buffering (trees built every step)
    - vLLM-backed tree expansion with live post-step reload
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        args: Optional[TreeRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
    ):
        if args is None:
            raise ValueError("TreeRPOConfig is required")

        # --- load model ---
        model_init_kwargs = args.model_init_kwargs or {}
        model_id: str
        if isinstance(model, str):
            model_id = model
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_init_kwargs,
            )
        else:
            model_id = getattr(model.config, "_name_or_path", "unknown-model")

        # Required for training
        if hasattr(model, "config"):
            model.config.use_cache = False

        # tokenizer
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                padding_side="left",
                trust_remote_code=args.trust_remote_code,
            )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.beta = args.beta
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        self.max_elements_per_forward = args.max_elements_per_forward

        # Metrics store
        self._metrics: Dict[str, Dict[str, List[float]]] = {
            "train": defaultdict(list),
            "eval": defaultdict(list),
        }

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # seed (note: CLI doesn’t expose it; we still honor args.seed default)
        set_seed_all(args.seed)

        # ---------- vLLM engine (single process) ----------
        self.vllm_engine: Optional[AsyncLLMEngine] = None

        if args.use_vllm:
            # Ensure a local directory that vLLM will read from on reloads.
            self._vllm_model_dir = os.path.join(self.args.output_dir, "_vllm_model")
            _ensure_dir(self._vllm_model_dir)

            # Save current HF weights there so the engine can load immediately.
            unwrapped = self.accelerator.unwrap_model(self.model) if hasattr(self, "accelerator") else self.model
            unwrapped.save_pretrained(self._vllm_model_dir, safe_serialization=True)

            vllm_dtype = _to_vllm_dtype_str(getattr(args, "torch_dtype", None), self.model)

            engine_args = AsyncEngineArgs(
                model=self._vllm_model_dir,          # crucial: point to the local dir we (re)save into
                dtype=vllm_dtype,
                gpu_memory_utilization=args.vllm_gpu_mem_util,
                max_num_seqs=args.vllm_max_num_seqs,
                max_num_batched_tokens=args.vllm_max_batched_tokens,
                trust_remote_code=args.trust_remote_code,
                enable_prefix_caching=True,
            )
            self.vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
            self.vllm_engine.log_requests = False

        if self.vllm_engine is None:
            raise ValueError("TreeRPOTrainer currently assumes use_vllm=True for tree building.")

        # Tree builder
        self.tree_builder = TreeBuilder(
            engine=self.vllm_engine,
            tokenizer=self.tokenizer,
            cfg=args,  # TreeRPOConfig
        )

    # -------------------- core loop overrides -------------------- #

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_closure=None,
        on_tpu: bool = False,
        using_lbfgs: bool = False,
    ):
        """Standard optimizer step, then reload vLLM weights once per step."""
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure, on_tpu, using_lbfgs)
        self._reload_weights_in_vllm()

    def _reload_weights_in_vllm(self):
        """Save updated HF weights and ask vLLM to reload them."""
        if self.vllm_engine is None:
            return

        # 1) save updated HF weights into the engine's model dir
        unwrapped = self.accelerator.unwrap_model(self.model) if hasattr(self, "accelerator") else self.model
        unwrapped.save_pretrained(self._vllm_model_dir, safe_serialization=True)

        # 2) get an engine client exposing the needed RPCs
        client = _find_vllm_client(self.vllm_engine)
        if client is None:
            # Best effort: cannot live-reload on this build
            return

        # 3) reset cache → reload → reset cache
        try:
            _run_coro_blocking(client.reset_prefix_cache())
        except Exception:
            pass

        try:
            _run_coro_blocking(client.collective_rpc("load_model"))
        except Exception as e:
            # If your vLLM build uses a different opcode (e.g., "reload_model"), adapt here.
            raise e

        try:
            _run_coro_blocking(client.reset_prefix_cache())
        except Exception:
            pass

        torch.cuda.empty_cache()

    # -------------------- data → trees --------------------------- #

    def _extract_batch_examples(self, inputs: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Normalize a Trainer batch into a list of {'question','final_answer'} dicts."""
        if isinstance(inputs, list):
            return inputs  # already list of examples
        if not isinstance(inputs, dict):
            raise TypeError(f"Unexpected batch type: {type(inputs)}")

        # Common case: dict of columns -> lists
        questions = inputs.get("question", None)
        answers = inputs.get("final_answer", None)
        if questions is None or answers is None:
            raise KeyError("Batch must contain 'question' and 'final_answer' fields.")

        # If tensors slipped in, convert to list of python objects
        if torch.is_tensor(questions):
            questions = [q for q in questions]
        if torch.is_tensor(answers):
            answers = [a for a in answers]

        # Ensure strings
        return [{"question": str(q), "final_answer": str(a)} for q, a in zip(questions, answers)]

    def _count_leaves(self, root) -> int:
        """Count leaves (nodes with no children) from a TreeNode root."""
        stack = [root]
        c = 0
        while stack:
            n = stack.pop()
            if n.children:
                stack.extend(n.children)
            else:
                c += 1
        return c

    def _prepare_inputs(self, inputs: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        """Build a tree for each prompt in the batch and convert to sibling groups."""
        mode = "eval" if self.control.should_evaluate else "train"
        examples = self._extract_batch_examples(inputs)
        problems = [ex["question"] for ex in examples]
        answers = [ex["final_answer"] for ex in examples]

        # Build trees concurrently
        trees = _run_coro_blocking(asyncio.gather(*[
            self.tree_builder.expand_tree(p, a) for p, a in zip(problems, answers)
        ]))

        # Convert each tree into a list of group dicts
        batch_groups = [self._convert_tree_to_training_inputs(t) for t in trees]

        # Minimal metrics (optional)
        if getattr(self.args, "log_tree_stats", False):
            leaves = [self._count_leaves(t) for t in trees]
            if leaves:
                avg_leaves = sum(leaves) / len(leaves)
                self._metrics[mode]["tree/avg_leaves"].append(float(avg_leaves))

        return batch_groups  # list (batch) of list (groups)

    def _convert_tree_to_training_inputs(self, tree_root) -> List[Dict[str, torch.Tensor]]:
        """
        Turn each sibling set (>=2 children) into a training 'group' dict.
        Advantage = child_reward - mean(sibling_rewards).
        """
        pad_id = self.tokenizer.pad_token_id
        groups: List[Dict[str, torch.Tensor]] = []

        def process_node(node):
            if len(node.children) < 2:
                return
            child_nodes = [c for c in node.children if c.reward is not None]
            if len(child_nodes) < 2:
                return

            rewards = torch.tensor([float(c.reward) for c in child_nodes], dtype=torch.float32)
            mean_r = rewards.mean()
            advantages = rewards - mean_r  # no std scaling (Dr-GRPO style)

            # pad prompts & completions separately
            prompt_tensors = [torch.tensor(c.prompt_ids, dtype=torch.long) for c in child_nodes]
            compl_tensors = [torch.tensor(c.completion_ids, dtype=torch.long) for c in child_nodes]

            prompt_ids = pad_1d(prompt_tensors, padding_value=pad_id)
            completion_ids = pad_1d(compl_tensors, padding_value=pad_id)

            prompt_mask = (prompt_ids != pad_id).long()
            completion_mask = (completion_ids != pad_id).long()

            groups.append({
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask,
                "completion_ids": completion_ids,
                "completion_mask": completion_mask,
                "advantages": advantages,
                "old_per_token_logps": None,   # μ=1 for now; could cache across iters if needed
                "ref_per_token_logps": None,   # beta==0 by default
            })

            for c in child_nodes:
                process_node(c)

        process_node(tree_root)
        return groups

    # -------------------- losses --------------------------- #

    def _get_per_token_logps(self, model, input_ids, attention_mask, completion_len: int):
        """
        Compute per-token log-probs for the *completion* span only.
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, T, V]

        # standard next-token shift
        logits = logits[:, :-1, :]                 # [B, T-1, V]
        targets = input_ids[:, 1:]                 # [B, T-1]

        # slice to the last `completion_len` positions
        comp_logits = logits[:, -completion_len:, :]         # [B, C, V]
        comp_targets = targets[:, -completion_len:]          # [B, C]

        log_probs = torch.log_softmax(comp_logits, dim=-1)   # [B, C, V]
        per_token_logps = log_probs.gather(-1, comp_targets.unsqueeze(-1)).squeeze(-1)  # [B, C]
        return per_token_logps

    def _compute_loss_for_group(self, model, group: Dict[str, torch.Tensor]) -> torch.Tensor:
        prompt_ids, prompt_mask = group["prompt_ids"], group["prompt_mask"]
        completion_ids, completion_mask = group["completion_ids"], group["completion_mask"]

        # concat to form full inputs
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(model.device)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1).to(model.device)
        completion_len = completion_ids.size(1)

        # row-wise fallback to avoid OOM
        total_elems = input_ids.shape[0] * input_ids.shape[1]
        if total_elems > self.max_elements_per_forward:
            chunks = []
            for i in range(input_ids.size(0)):
                lp = self._get_per_token_logps(
                    model, input_ids[i:i+1], attention_mask[i:i+1], completion_len
                )
                chunks.append(lp)
            per_token_logps = torch.cat(chunks, dim=0)
        else:
            per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, completion_len)

        # clipped surrogate (no KL by default)
        advantages = group["advantages"].to(model.device)                       # [B]
        old_per_token_logps = group["old_per_token_logps"] or per_token_logps.detach()
        log_ratio = (per_token_logps - old_per_token_logps).clamp(-60, 60)     # [B, C]
        ratio = torch.exp(log_ratio)
        ratio_clipped = torch.clamp(ratio, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # broadcast advantages to token level
        adv = advantages.unsqueeze(1)                                           # [B, 1]
        loss1 = ratio * adv
        loss2 = ratio_clipped * adv
        per_token_loss = -torch.min(loss1, loss2)                               # [B, C]

        # mask and average over completion tokens
        cmask = completion_mask.to(model.device)                                # [B, C]
        loss = (per_token_loss * cmask).sum() / cmask.sum().clamp_min(1)
        return loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Any], num_items_in_batch=None) -> torch.Tensor:
        """Build trees for the batch, compute group losses, mean over all groups."""
        model.train()
        batch_groups: List[List[Dict[str, torch.Tensor]]] = self._prepare_inputs(inputs)

        group_losses: List[torch.Tensor] = []
        for groups in batch_groups:
            if not groups:
                continue
            # Option: scale per-prompt; we keep equal weighting across groups
            for g in groups:
                loss = self._compute_loss_for_group(model, g)
                self.accelerator.backward(loss)
                group_losses.append(loss.detach())

        return (torch.stack(group_losses).mean()
                if group_losses
                else torch.zeros(1, device=self.accelerator.device))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Not used; we override training_step. Keep for eval compatibility if needed.
        return torch.zeros(1, device=self.accelerator.device)

    # -------------------- logging --------------------------- #

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {k: (sum(v) / len(v)) for k, v in self._metrics[mode].items() if v}
        logs = {**logs, **({f"{mode}_{k}": v for k, v in metrics.items()})}
        super().log(logs, start_time)
        self._metrics[mode].clear()
