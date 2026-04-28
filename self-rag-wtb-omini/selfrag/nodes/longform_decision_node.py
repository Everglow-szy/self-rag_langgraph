"""Long-form retrieval decision node.

Ports the decision logic from run_long_form_static.py:call_model_beam_batch (lines 159-189).
"""
from typing import Dict

import numpy as np
from vllm import SamplingParams


def build_longform_decision_node(model, ret_tokens: Dict[str, int], config):
    """Factory: returns a node that decides whether to retrieve for long-form generation."""

    def longform_decision_node(state):
        mode = state.get("mode", config.mode)

        if mode == "no_retrieval":
            return {"do_retrieve": False}
        if mode == "always_retrieve":
            return {"do_retrieve": True}

        # Adaptive: generate short prediction and check for [Retrieval] token
        sp = SamplingParams(
            temperature=0.0, top_p=1.0,
            max_tokens=25, logprobs=32000,
        )
        preds = model.generate([state["prompt"]], sp)
        pred_text = preds[0].outputs[0].text.split("\n\n")[0]
        pred_logprobs = preds[0].outputs[0].logprobs

        if "[Retrieval]" not in pred_text:
            return {"do_retrieve": False}

        threshold = state.get("threshold", config.threshold)
        if threshold is None:
            return {"do_retrieve": False}

        ret_score = {}
        for tok, tok_id in ret_tokens.items():
            prob = pred_logprobs[0][tok_id] if tok_id in pred_logprobs[0] else -100
            ret_score[tok] = float(np.exp(prob))
        denom = ret_score["[Retrieval]"] + ret_score["[No Retrieval]"]
        if denom > 0:
            retrieve_prob = ret_score["[Retrieval]"] / denom
        else:
            retrieve_prob = 0.0

        do_retrieve = retrieve_prob > threshold
        return {"do_retrieve": bool(do_retrieve)}

    return longform_decision_node