"""Generation nodes: evidence-based and no-retrieval generation with scoring.

Faithfully ports the scoring formulas from
``run_short_form.py:call_model_rerank_w_scores_batch`` lines 85-200.
"""
from typing import Any, Dict, List, Optional

import numpy as np
from vllm import SamplingParams


def build_evidence_generation_node(
    model,
    rel_tokens: Dict[str, int],
    grd_tokens: Optional[Dict[str, int]],
    ut_tokens: Optional[Dict[str, int]],
    config,
):
    """Factory: node that generates answers for each retrieved passage and scores them."""

    def evidence_generation_node(state):
        prompt = state["prompt"]
        passages = state.get("retrieved_passages") or state.get("evidences") or []
        if not passages:
            return {**state, "evidence_results": {}, "do_retrieve": False}

        ndocs = state.get("ndocs", config.ndocs)
        passages = passages[:ndocs]

        augmented = [
            prompt + "[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(
                p.get("title", ""), p.get("text", "")
            )
            for p in passages
        ]
        sp = SamplingParams(
            temperature=0.0, top_p=1.0,
            max_tokens=state.get("max_new_tokens", config.max_new_tokens),
            logprobs=5000,
        )
        preds = model.generate(augmented, sp)

        w_rel = state.get("w_rel", config.w_rel)
        w_sup = state.get("w_sup", config.w_sup)
        w_use = state.get("w_use", config.w_use)
        use_seq = state.get("use_seqscore", config.use_seqscore)

        results: Dict[str, Dict[str, Any]] = {}
        for p_idx, pred in enumerate(preds):
            token_ids = pred.outputs[0].token_ids
            text = pred.outputs[0].text
            logprobs = pred.outputs[0].logprobs
            seq_score = pred.outputs[0].cumulative_logprob / max(len(token_ids), 1)

            # ---- ISREL (relevance) ----
            rel_dict: Dict[str, float] = {}
            for tok, tid in rel_tokens.items():
                prob = logprobs[0][tid] if tid in logprobs[0] else -100
                rel_dict[tok] = float(np.exp(float(prob)))
            rel_sum = float(np.sum(list(rel_dict.values()))) or 1.0
            relevance_score = rel_dict["[Relevant]"] / rel_sum

            # ---- ISSUP (grounding) ----
            grd_dict: Dict[str, float] = {}
            if grd_tokens is not None:
                grd_vals = list(grd_tokens.values())
                grd_idx = None
                for i, tid in enumerate(token_ids):
                    if tid in grd_vals:
                        grd_idx = i
                        break
                if grd_idx is not None:
                    for tok, tid in grd_tokens.items():
                        prob = logprobs[grd_idx][tid] if tid in logprobs[grd_idx] else -100
                        grd_dict[tok] = float(np.exp(float(prob)))
            if len(grd_dict) == 3:
                gt_sum = float(np.sum(list(grd_dict.values()))) or 1.0
                ground_score = (grd_dict["[Fully supported]"] / gt_sum) + 0.5 * (
                    grd_dict["[Partially supported]"] / gt_sum
                )
            else:
                ground_score = 0.0

            # ---- ISUSE (utility) ----
            ut_dict: Dict[str, float] = {}
            if ut_tokens is not None:
                ut_vals = list(ut_tokens.values())
                # Original code uses FIRST occurrence (index [0])
                ut_indices = []
                for i, tid in enumerate(token_ids):
                    if tid in ut_vals:
                        ut_indices.append(i)
                if len(ut_indices) > 0:
                    idx = ut_indices[0]
                    for tok, tid in ut_tokens.items():
                        prob = logprobs[idx][tid] if tid in logprobs[idx] else -100
                        ut_dict[tok] = float(np.exp(float(prob)))
            if len(ut_dict) == 5:
                ut_sum = float(np.sum(list(ut_dict.values()))) or 1.0
                ut_scale = [-1, -0.5, 0, 0.5, 1]
                utility_score = float(np.sum([
                    ut_scale[i] * (ut_dict["[Utility:{}]".format(i + 1)] / ut_sum)
                    for i in range(5)
                ]))
            else:
                utility_score = 0.0

            if use_seq:
                final = (float(np.exp(seq_score))
                         + w_rel * relevance_score
                         + w_sup * ground_score
                         + w_use * utility_score)
            else:
                final = (w_rel * relevance_score
                         + w_sup * ground_score
                         + w_use * utility_score)

            results["retrieval_{}".format(p_idx)] = {
                "pred": text,
                "score": float(final),
                "ctx": passages[p_idx],
            }
        return {**state, "evidence_results": results}

    return evidence_generation_node


def build_no_retrieval_generation_node(model, config):
    """Factory: node that generates without retrieval (appends [No Retrieval])."""

    def no_retrieval_generation_node(state):
        sp = SamplingParams(
            temperature=0.0, top_p=1.0,
            max_tokens=state.get("max_new_tokens", config.max_new_tokens),
        )
        preds = model.generate([state["prompt"] + "[No Retrieval]"], sp)
        text = preds[0].outputs[0].text
        return {
            **state,
            "no_retrieval_pred": text,
            "evidence_results": state.get("evidence_results", {}),
        }

    return no_retrieval_generation_node
