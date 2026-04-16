"""Long-form prompt node: builds prompt with task instructions for ASQA/ELI5/FactScore."""
from utils import PROMPT_DICT, TASK_INST


def build_longform_prompt_node():
    """Factory: returns a node that constructs the prompt with task-specific instructions."""

    def longform_prompt_node(state):
        question = state["question"]
        task = state.get("task", "asqa")
        instructions = TASK_INST.get(task, "")
        full_input = instructions + "## Input:\n\n" + question
        prompt = PROMPT_DICT["prompt_no_input"].format_map({"instruction": full_input})
        return {**state, "prompt": prompt}

    return longform_prompt_node
