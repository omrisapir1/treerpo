# treerpo/utils/prompts.py
from typing import List
from transformers import AutoTokenizer

def build_prompt_ids(tok: AutoTokenizer, problem: str) -> List[int]:
    """Build token IDs using the model's chat template (user-only message)."""
    return tok.apply_chat_template(
        [{"role": "user", "content": problem}],
        tokenize=True,
        add_generation_prompt=True,
        continue_final_message=False,
    )
