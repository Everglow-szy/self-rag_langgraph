"""Decision node: adaptive retrieval decision via [Retrieval] token logprobs."""
from typing import Dict

from vllm import SamplingParams


def build_decision_node(model, ret_tokens: Dict[str, int], config):
    """Factory: returns a node that decides whether to retrieve.

    Faithfully ports the retrieval-decision logic from
    ``run_short_form.py:call_model_rerank_w_scores_batch`` lines 56-83.
    """

    def decision_node(state):
        mode = state.get("mode", config.mode)

        if mode == "always_retrieve":
            return {"do_retrieve": True}
        if mode == "no_retrieval":
            return {"do_retrieve": False}

        # Adaptive: generate initial prediction and inspect logprobs
        sp = SamplingParams(
            temperature=0.0, top_p=1.0,
            max_tokens=state.get("max_new_tokens", config.max_new_tokens),
            logprobs=32016,
        )
        preds = model.generate([state["prompt"]], sp)
        first_logprobs = preds[0].outputs[0].logprobs

        score: Dict[str, float] = {}
        for tok, tok_id in ret_tokens.items():
            if tok_id not in first_logprobs[0]:
                score[tok] = -100.0
            else:
                score[tok] = float(first_logprobs[0][tok_id])

        thr = state.get("threshold", config.threshold)
        do_retrieve = (
            score["[Retrieval]"]
            / (score["[Retrieval]"] + score["[No Retrieval]"])
            > thr
        )
        return {
            "do_retrieve": bool(do_retrieve),
            "retrieval_decision_scores": score,
        }

    return decision_node