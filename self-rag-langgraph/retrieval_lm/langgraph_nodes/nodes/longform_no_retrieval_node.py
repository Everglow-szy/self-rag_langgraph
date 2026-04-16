"""Long-form no-retrieval generation node.

Ports the no-retrieval path from run_long_form_static.py:call_model_beam_batch (lines 191-198).
"""
from vllm import SamplingParams


def build_longform_no_retrieval_node(model, config):
    """Factory: node that generates without retrieval for long-form tasks."""

    def longform_no_retrieval_node(state):
        sp = SamplingParams(
            temperature=0.0, top_p=1.0,
            max_tokens=state.get("max_new_tokens", config.max_new_tokens),
        )
        preds = model.generate([state["prompt"] + "[No Retrieval]"], sp)
        text = preds[0].outputs[0].text.split("\n\n")[0]
        return {**state, "no_retrieval_pred": text}

    return longform_no_retrieval_node
