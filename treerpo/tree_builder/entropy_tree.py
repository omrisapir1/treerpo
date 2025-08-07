# ======================================================================
# TreeRPO — Refactored Implementation (async, vLLM-compatible)
# ======================================================================
from __future__ import annotations

import asyncio
import uuid
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine


from treerpo.utils.answers import extract_final_answer, math_equal
from treerpo.utils.prompts import build_prompt_ids
from treerpo.config import TreeRPOConfig as TreeConfig


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
class NodeState(Enum):
    EXPLORING = auto()
    TERMINAL = auto()


@dataclass
class TreeNode:
    # Context up to this node (prompt IDs)
    prompt_ids: List[int]

    # Hierarchy
    depth: int = 0
    parent: Optional["TreeNode"] = None
    children: List["TreeNode"] = field(default_factory=list)

    # Local “thought span” (this node’s continuation)
    completion_ids: List[int] = field(default_factory=list)

    # Text caches (filled on finalize)
    prompt_text: str = ""
    completion_text: str = ""

    # Bookkeeping
    state: NodeState = NodeState.EXPLORING
    reward: Optional[float] = None                 # leaf: {0,1}; interior: mean of descendants
    rewards_accum: List[float] = field(default_factory=list)  # descendant leaf rewards

    def add_child(self, child: "TreeNode") -> None:
        self.children.append(child)

    def propagate_reward_up(self, r: float) -> None:
        """Push a leaf reward to ancestors (called after leaf scoring)."""
        self.rewards_accum.append(r)
        if self.parent is not None:
            self.parent.propagate_reward_up(r)

    def finalize_interior_reward(self) -> None:
        """Set interior reward as mean of accumulated leaf rewards."""
        if self.state is NodeState.TERMINAL:
            return
        if self.rewards_accum:
            self.reward = float(np.mean(self.rewards_accum))
        else:
            self.reward = 0.0


# ----------------------------------------------------------------------
# Main builder
# ----------------------------------------------------------------------
class TreeBuilder:
    def __init__(self, engine: AsyncLLMEngine, tokenizer: AutoTokenizer, cfg: TreeConfig) -> None:
        self.engine = engine
        self.tok = tokenizer
        self.cfg = cfg
        self.sem = asyncio.Semaphore(cfg.max_async_streams)

        # Seeding (best effort; vLLM may have its own RNG)
        if cfg.seed is not None:
            random.seed(cfg.seed)
            np.random.seed(cfg.seed)

    # ---------------------------- Public API ----------------------------
    async def expand_tree(self, problem: str, final_answer: str) -> TreeNode:
        """Build a full reasoning tree for a single prompt."""
        root = TreeNode(prompt_ids=build_prompt_ids(self.tok, problem), depth=0, parent=None)

        # Maintain a task list scoped to this call (avoids cross-call races)
        tasks: list[asyncio.Task] = []
        await self._spawn_node(root, final_answer=final_answer, coverage_mode=False, tasks=tasks)

        # Wait for all spawned tasks (children/grandchildren)
        while True:
            pending = [t for t in tasks if not t.done()]
            if not pending:
                break
            await asyncio.gather(*pending)

        # Reward propagation: leaves already have reward; push up then set interior means
        for n in self._collect_nodes(root):
            if n.state is NodeState.TERMINAL:
                n.propagate_reward_up(n.reward or 0.0)
        for n in self._collect_nodes(root):
            n.finalize_interior_reward()

        return root

    # ---------------------------- Node growth ---------------------------
    async def _spawn_node(
        self,
        node: TreeNode,
        *,
        final_answer: str,
        coverage_mode: bool,
        tasks: list[asyncio.Task],
    ) -> None:
        """
        Grow a node into either interior (split) or leaf (terminate).
        If coverage_mode is True: generate to leaf with NO further splitting.
        """
        async with self.sem:
            params = SamplingParams(
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                top_k=self.cfg.top_k,
                repetition_penalty=self.cfg.repetition_penalty,
                max_tokens=self.cfg.max_completion_length,
                logprobs=self.cfg.entropy_top_k,  # compute entropy over top-K
            )

            prompt_text = self.tok.decode(node.prompt_ids)
            request_id = str(uuid.uuid4())

            did_split = False
            last_outs = None

            # Stream tokens from vLLM
            try:
                token_idx_processed = 0
                prev_token_was_delim = False  # "just after delimiter" = prev=True and current not delimiter

                async for chunk in self.engine.generate(prompt_text, params, request_id=request_id):
                    outs = chunk.outputs[0]
                    last_outs = outs  # keep last for coverage decisions

                    # Depth limit: depth >= max_depth means NO MORE splits, but still generate to leaf.
                    allow_splitting = (not coverage_mode) and (node.depth < self.cfg.max_depth)

                    # Iterate new token positions since last iteration
                    total = len(outs.token_ids)
                    if total <= token_idx_processed:
                        continue

                    # Update completion_ids incrementally
                    new_token_ids = outs.token_ids[token_idx_processed:total]
                    node.completion_ids.extend(new_token_ids)

                    # Update delimiter detector (token-level approximation)
                    decoded_now = self.tok.decode(node.completion_ids)
                    now_ends_with_delim = decoded_now[-1] in self.cfg.split_delimiters if decoded_now else False
                    just_after_delim = (prev_token_was_delim and not now_ends_with_delim)


                    forced_root = self.cfg.forced_root_split and (node.depth == 0 and token_idx_processed == 0)

                    # Consider splitting if allowed
                    if allow_splitting and (forced_root or (
                        self._segment_long_enough(node) and just_after_delim
                        and self._entropy_from_topk_logprobs(outs, at_index=total - 1) >= self.cfg.entropy_threshold
                    )):
                        # Prepare parent to end EXACTLY at the split boundary:
                        if node.completion_ids:
                            first_token_for_child = node.completion_ids.pop()  # original token
                        else:
                            first_token_for_child = outs.token_ids[total - 1]

                        # Abort parent stream to free resources
                        await self.engine.abort(request_id=request_id)

                        # Parent's fixed text contexts
                        parent_prompt_ids = node.prompt_ids
                        parent_completion_ids = node.completion_ids
                        shared_prefix = parent_prompt_ids + parent_completion_ids  # children start from here

                        # Compute alternative first token (exclude original)
                        top_map = outs.logprobs[total - 1] if (outs.logprobs and total - 1 in outs.logprobs) else None
                        if not top_map:
                            alt_token = first_token_for_child
                        else:
                            cand = [(tid, lp.logprob) for tid, lp in top_map.items() if tid != first_token_for_child]
                            if not cand:
                                alt_token = first_token_for_child
                            else:
                                cand_ids, cand_lps = zip(*cand)
                                logits = np.array(cand_lps, dtype=float)
                                probs = np.exp(logits / max(self.cfg.temperature, 1e-8))
                                probs /= probs.sum()
                                alt_token = int(np.random.choice(cand_ids, p=probs))

                        # Create exactly two children
                        for t0 in (first_token_for_child, alt_token):
                            child = TreeNode(prompt_ids=shared_prefix + [t0], depth=node.depth + 1, parent=node)
                            node.add_child(child)
                            self._schedule_child(child, final_answer=final_answer, coverage_mode=False, tasks=tasks)

                        did_split = True
                        break  # stop processing this node; children continue

                    # Update delimiter tracker for next step
                    prev_token_was_delim = now_ends_with_delim
                    token_idx_processed = total

            finally:
                # If split happened, we already aborted the stream and scheduled children
                pass

            if did_split:
                return

            # --- No split happened during streaming ---
            # If we get here, the node generated to EOS or max tokens.
            if (not coverage_mode) and self._maybe_should_add_coverage(node, last_outs):
                # Create up to N coverage children at the last safe delimiter
                # and let them CONTINUE GENERATION but with splitting DISABLED.
                self._spawn_coverage_children(node, last_outs, final_answer=final_answer, tasks=tasks)
                return

            # Otherwise: finalize as a leaf and score
            self._finalize_leaf(node)
            node.reward = self._score_leaf(node, final_answer)

    # ---------------------------- Coverage helpers ----------------------
    def _maybe_should_add_coverage(self, node: TreeNode, outs) -> bool:
        """Decide whether to perform a post-hoc coverage split."""
        if outs is None:
            return False
        # Require at least S_min chars after the split point
        text = outs.text or self.tok.decode(node.completion_ids)
        if len(text) < self.cfg.coverage_min_chars:
            return False

        # Find last delimiter occurring ≥ S_min chars before the end
        cutoff = len(text) - self.cfg.coverage_min_chars
        last_idx = self._last_delimiter_before(text, cutoff)
        if last_idx < 0:
            return False

        # Also require segment length ≥ L_min tokens when truncated to that delimiter
        truncated = text[: last_idx + 1]
        trunc_ids = self.tok.encode(truncated, add_special_tokens=False)
        if len(trunc_ids) < self.cfg.min_segment_len:
            return False

        # Cache truncation on node for spawn_coverage_children
        node._coverage_trunc_ids = trunc_ids   # attach ephemeral attribute
        node._coverage_shared_prefix = node.prompt_ids + trunc_ids
        return True

    def _spawn_coverage_children(self, node: TreeNode, outs, *, final_answer: str, tasks: list[asyncio.Task]) -> None:
        """Spawn up to cfg.coverage_children_max children from the coverage split point."""
        shared_prefix = node._coverage_shared_prefix  # set in _maybe_should_add_coverage
        n_children = int(self.cfg.coverage_children_max)
        for _ in range(n_children):
            child = TreeNode(prompt_ids=list(shared_prefix), depth=node.depth + 1, parent=node)
            node.add_child(child)
            # coverage_mode=True => generate to leaf; NO splitting or re-coverage
            self._schedule_child(child, final_answer=final_answer, coverage_mode=True, tasks=tasks)

    # ---------------------------- Scheduling ----------------------------
    def _schedule_child(self, child: TreeNode, *, final_answer: str, coverage_mode: bool, tasks: list[asyncio.Task]) -> None:
        tasks.append(asyncio.create_task(self._spawn_node(child, final_answer=final_answer, coverage_mode=coverage_mode, tasks=tasks)))

    # ---------------------------- Termination & scoring -----------------
    def _finalize_leaf(self, node: TreeNode) -> None:
        node.state = NodeState.TERMINAL
        node.prompt_text = self.tok.decode(node.prompt_ids)
        node.completion_text = self.tok.decode(node.completion_ids)

    def _score_leaf(self, node: TreeNode, final_answer: str) -> float:
        pred = extract_final_answer(node.completion_text)
        try:
            return float(int(math_equal(pred, final_answer)))
        except Exception:
            return 0.0

    # ---------------------------- Predicates & utilities ----------------
    def _segment_long_enough(self, node: TreeNode) -> bool:
        return len(node.completion_ids) >= self.cfg.min_segment_len

    def _entropy_from_topk_logprobs(self, outs, at_index: int) -> float:
        """Shannon entropy from vLLM's logprobs dict at a given token index (top-K only)."""
        if outs is None or not outs.logprobs or at_index not in outs.logprobs:
            return 0.0
        top_map = outs.logprobs[at_index]
        if not top_map or len(top_map) < 2:
            return 0.0
        # vLLM returns {token_id: Logprob(token, logprob, rank, ...)}
        vals = np.array([entry.logprob for entry in top_map.values()], dtype=float)
        # normalize within top-K
        p = np.exp(vals - vals.max())
        p /= p.sum()
        return float(-(p * np.log(p + 1e-12)).sum())

    def _last_delimiter_before(self, s: str, cutoff: int) -> int:
        """Return last index in s[:cutoff] that is a delimiter; -1 if none."""
        for i in range(min(cutoff, len(s) - 1), -1, -1):
            if s[i] in self.cfg.split_delimiters:
                return i
        return -1

    def _collect_nodes(self, root: TreeNode) -> List[TreeNode]:
        out: List[TreeNode] = []
        st = [root]
        while st:
            n = st.pop()
            out.append(n)
            st.extend(n.children)
        return out
