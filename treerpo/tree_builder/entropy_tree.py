import asyncio
import json
import os
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

import numpy as np
from transformers import AutoTokenizer
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

    # Local "thought span" (this node's continuation)
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

    def to_dict(self):
        return {
            "prompt_ids": self.prompt_ids,
            "completion_ids": self.completion_ids,
            "prompt_text": self.prompt_text,
            "completion_text": self.completion_text,
            "reward": self.reward,
            "rewards": self.rewards_accum,
            "depth": self.depth,
            "children": [c.to_dict() for c in self.children],

        }


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

        # Start with the root node - the unified _spawn_node handles the forced first split
        tasks: list[asyncio.Task] = []
        await self._spawn_node(root, final_answer=final_answer, coverage_mode=False, tasks=tasks)

        # Wait for all spawned tasks (matching original ToE logic)
        if tasks:
            await asyncio.gather(*tasks)

            while True:
                # Keep only tasks that are not finished yet
                pending = [t for t in tasks if not t.done()]
                if not pending:
                    break
                # Await the current batch; new tasks may be appended meanwhile
                await asyncio.gather(*pending)

        # Reward propagation: leaves already have reward; push up then set interior means
        for n in self._collect_nodes(root):
            if n.state is NodeState.TERMINAL:
                n.propagate_reward_up(n.reward or 0.0)
        for n in self._collect_nodes(root):
            n.finalize_interior_reward()

        js = root.to_dict()
        js['final_answer'] = final_answer
        path = '/workspace/data'
        if not os.path.exists(path):
            os.mkdir(path)
        import time
        json.dump( js, open(os.path.join(path, f"{time.time()}.json"),'w'))


        return root


    # ---------------------------- Node growth ---------------------------
    async def _spawn_node(
        self,
        node: TreeNode,
        *,
        final_answer: str,
        coverage_mode: bool,
        n_tokens_generated: int = 0,
        tasks: list[asyncio.Task],
    ) -> None:
        """
        Unified node spawning method that handles all splitting logic inline.
        Based on the original ToE._spawn method structure.
        """
        async with self.sem:
            from vllm import SamplingParams
            import uuid

            # Set up sampling parameters
            params = SamplingParams(
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                top_k=self.cfg.top_k,
                repetition_penalty=self.cfg.repetition_penalty,
                max_tokens=min(self.cfg.max_full_answer_length - n_tokens_generated, self.cfg.max_completion_length) ,
                logprobs=self.cfg.entropy_top_k,
            )

            # Generate text for this node
            prompt_text = self.tok.decode(node.prompt_ids)
            token_counter = 0
            at_splitable_token = False

            async for chunk in self.engine.generate(prompt_text, params, request_id=str(uuid.uuid4())):
                if coverage_mode:
                    continue  # Skip splitting logic in coverage mode

                outs = chunk.outputs[0]
                total_tokens = len(outs.token_ids)

                # Skip if we don't have enough tokens yet (except for root)
                if node.depth > 0 and total_tokens < self.cfg.min_segment_len:
                    continue

                # Check entropy at each new token position
                for token_index in range(token_counter, total_tokens):
                    out_tokens = outs.token_ids[:token_index + 1]
                    out_text = self.tok.decode(out_tokens)

                    # For depth 0: split immediately on first token (no conditions)
                    # For depth > 0: check entropy and other conditions
                    top_candidates= {tid: -99 if np.isinf(e.logprob) else e.logprob for tid, e in
                           outs.logprobs[token_index].items()} if outs.logprobs else {}
                    if node.depth == 0:
                        should_split = True
                    else:
                        # Calculate entropy from logprobs (only for non-root splits)
                        entropy = self._entropy_from_topk_logprobs(top_candidates)

                        # Entropy-based split conditions
                        should_split = (
                            entropy > self.cfg.entropy_threshold and
                            node.depth < self.cfg.max_depth and
                            at_splitable_token and
                            len(out_tokens) >= self.cfg.min_segment_len and
                            out_text and out_text[-1] not in self.cfg.split_delimiters
                        )

                    # For depth > 0, we need to set these variables for the split logic
                    if node.depth > 0:
                        out_tokens = outs.token_ids[:token_index + 1]
                        out_text = self.tok.decode(out_tokens)

                    # UNIFIED SPLITTING DECISION: entropy-based OR forced first split
                    if should_split:
                        # Update this node with completion up to the split point
                        take_from_prompt = (node.depth != 0)
                        last_token_id = self._update_node_with_output(
                            node, out_tokens, take_from_prompt, remove_last_token=True
                        )


                        # Prepare next prompt for children
                        next_prompt_ids = node.prompt_ids + node.completion_ids

                        # Get alternative tokens for diversification (matching original ToE.py exactly)
                        # if outs.logprobs and token_index in outs.logprobs:
                            # Preprocess logprobs like original ToE.py

                            # Get candidates excluding the original token (matching original ToE.py)
                        cand_items = [(t, lp) for t, lp in top_candidates.items() if t != last_token_id]


                        if cand_items:
                            cand_ids, cand_lps = zip(*cand_items)
                            probs = np.exp(np.array(cand_lps) / self.cfg.temperature)
                            probs = probs / probs.sum()
                            alt_token_id = int(np.random.choice(cand_ids, p=probs))
                        else:
                            alt_token_id = last_token_id  # Fallback
                        # else:
                        #     alt_token_id = last_token_id  # Fallback

                        # Create two children: original + alternative (matching original ToE.py)
                        for forced_token in (last_token_id, alt_token_id):

                            child = TreeNode(
                                prompt_ids=next_prompt_ids + [forced_token],
                                depth=node.depth + 1,
                                parent=node
                            )
                            node.add_child(child)
                            n_tokens_generated += total_tokens
                            self._schedule_child(child, final_answer=final_answer, coverage_mode=False, n_tokens_generated=n_tokens_generated, tasks=tasks)

                        return  # Stop this generation stream

                    # Update delimiter tracking
                    at_splitable_token = (out_text and out_text[-1] in self.cfg.split_delimiters) if out_text else False

                token_counter = total_tokens

            # Generation completed - handle coverage split or finalize
            final_output = chunk.outputs[0]

            # COVERAGE SPLIT LOGIC (matching original ToE)
            if not coverage_mode:
                # Look for coverage split opportunity
                split_point = self._last_delimiter_before(
                    final_output.text,
                    len(final_output.text) - self.cfg.coverage_min_chars
                )

                if (split_point != -1 and
                    split_point >= self.cfg.min_segment_len):

                    # Truncate this node at the split point
                    truncated_text = final_output.text[:split_point + 1]
                    truncated_ids = self.tok.encode(truncated_text, add_special_tokens=False)

                    take_from_prompt = (node.depth != 0)

                    self._update_node_with_output(
                        node, final_output.token_ids, take_from_prompt,
                        remove_last_token=False, new_completion_ids=truncated_ids
                    )

                    # Create coverage children
                    next_prompt_ids = node.prompt_ids + node.completion_ids
                    num_coverage_children = getattr(self.cfg, 'coverage_children_max', 4)

                    for _ in range(num_coverage_children):
                        child = TreeNode(
                            prompt_ids=next_prompt_ids,
                            depth=node.depth + 1,
                            parent=node
                        )
                        node.add_child(child)
                        n_tokens_generated += total_tokens
                        self._schedule_child(child, final_answer=final_answer, coverage_mode=True,n_tokens_generated=n_tokens_generated, tasks=tasks)

                    return

            # TERMINAL LEAF: finalize this node
            take_from_prompt = (node.depth != 0) if not coverage_mode else False
            self._update_node_with_output(
                node, final_output.token_ids, take_from_prompt, remove_last_token=False
            )
            self._finalize_leaf(node)
            node.reward = self._score_leaf(node, final_answer)

    def _update_node_with_output(self, node: TreeNode, output_tokens: list,
                                take_one_from_prompt: bool, remove_last_token: bool,
                                new_completion_ids=None) -> int:
        """Update node with generation output, matching original ToE logic."""
        last_token_id = None
        init_addition_tokens = []
        node.completion_ids = output_tokens
        if remove_last_token and node.completion_ids:
            last_token_id = node.completion_ids[-1]
            node.completion_ids = node.completion_ids[:-1]
        elif new_completion_ids:  # Match original ToE.py: use truthiness, not "is not None"
            node.completion_ids = new_completion_ids

        if take_one_from_prompt and node.prompt_ids:
            init_addition_tokens.append(node.prompt_ids[-1])
            node.prompt_ids = node.prompt_ids[:-1]
            node.completion_ids = init_addition_tokens + node.completion_ids

        node.prompt_text = self.tok.decode(node.prompt_ids)
        node.completion_text = self.tok.decode(node.completion_ids)
        return last_token_id

    # ---------------------------- Coverage helpers ----------------------
    def _maybe_should_add_coverage(self, node: TreeNode, outs) -> bool:
        """Decide whether to perform a post-hoc coverage split."""
        if not outs or not outs.text:
            return False

        # Check if the generated text is longer than the coverage threshold
        text_length = len(outs.text)
        if text_length <= self.cfg.coverage_min_chars:
            return False

        # Look for a natural split point (delimiter) before the minimum coverage threshold
        split_point = self._last_delimiter_before(outs.text, text_length - self.cfg.coverage_min_chars)

        # Only split if we found a delimiter and it's far enough from the start
        return split_point != -1 and split_point >= self.cfg.min_segment_len

    # ---------------------------- Scheduling ----------------------------
    def _schedule_child(self, child: TreeNode, *, final_answer: str, coverage_mode: bool, n_tokens_generated:int, tasks: list[asyncio.Task]) -> None:
        """Schedule a child node for asynchronous generation."""
        tasks.append(asyncio.create_task(
            self._spawn_node(child, final_answer=final_answer, coverage_mode=coverage_mode, n_tokens_generated=n_tokens_generated, tasks=tasks)
        ))

    # ---------------------------- Termination & scoring -----------------
    def _finalize_leaf(self, node: TreeNode) -> None:
        """Mark a node as terminal and decode its text."""
        node.state = NodeState.TERMINAL
        node.prompt_text = self.tok.decode(node.prompt_ids)
        node.completion_text = self.tok.decode(node.completion_ids)

    def _score_leaf(self, node: TreeNode, final_answer: str) -> float:
        """Score a leaf node based on whether its answer matches the reference."""
        pred = extract_final_answer(node.completion_text)
        try:
            return float(int(math_equal(pred, final_answer)))
        except Exception:
            return 0.0

    # ---------------------------- Predicates & utilities ----------------
    def _segment_long_enough(self, node: TreeNode) -> bool:
        """Check if the current completion segment is long enough to split."""
        return len(node.completion_ids) >= self.cfg.min_segment_len

    def _entropy_from_topk_logprobs(self, top_candidates: dict) -> float:
        """Entropy from vLLM's logprobs dict at a given token index (top-K only)."""
        if len(top_candidates) > 1:
            p = np.exp(list(top_candidates.values()));
            p /= p.sum()
            return float(-(p * np.log(p)).sum())
        return 0.0

    def _last_delimiter_before(self, s: str, cutoff: int) -> int:
        """Return last index in s[:cutoff] that is a delimiter; -1 if none."""
        for i in range(min(cutoff, len(s) - 1), -1, -1):
            if s[i] in self.cfg.split_delimiters:
                return i
        return -1

    def _collect_nodes(self, root: TreeNode) -> List[TreeNode]:
        """Collect all nodes in the tree in post-order."""
        out: List[TreeNode] = []
        st = [root]
        while st:
            n = st.pop()
            out.append(n)
            st.extend(n.children)
        return out
