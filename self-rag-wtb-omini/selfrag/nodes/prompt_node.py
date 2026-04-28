"""Prompt node: builds the Self-RAG instruction prompt from the question."""
from selfrag.constants import PROMPT_DICT


def build_prompt_node():
    """Factory: returns a node that constructs the Self-RAG prompt."""

    def prompt_node(state):
        q = state["question"]
        prompt = PROMPT_DICT["prompt_no_input"].format_map({"instruction": q})
        return {"prompt": prompt}

    return prompt_node