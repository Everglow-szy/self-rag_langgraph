"""Beam step node: one step of tree-based beam search generation.

Faithfully ports the core loop from ``run_long_form_static.py:call_model_beam_batch``
and uses ``run_step_generation_batch`` scoring logic.
"""
from typing import Any, Dict, List, Optional

import numpy as np
from vllm import SamplingParams


def build_beam_step_node(
    model,
    rel_tokens: Dict[str, int],
    grd_tokens: Optional[Dict[str, int]],
    ret_tokens: Dict[str, int],
    ut_tokens: Optional[Dict[str, int]],
    config,
):
    """Factory: returns a node that runs one depth level of beam search."""

    def _run_step_generation(prompt, paragraphs, max_new_tokens,
                             w_rel, w_sup, w_use, use_seqscore, threshold):
        """Port of run_step_generation_batch from run_long_form_static.py."""
        aug_prompts = [
            prompt + "[Retrieval]<paragraph>{}\n{}</paragraph>".format(
                p.get("title", ""), p.get("text", ""))
            for p in paragraphs
        ]

        sp = SamplingParams(
            temperature=0.0, top_p=1.0,
            max_tokens=max_new_tokens, logprobs=32000,
        )
        preds = model.generate(aug_prompts, sp)

        final_preds = []
        scores = []
        overall_scores = {}

        for p_idx, pred in enumerate(preds):
            token_ids = pred.outputs[0].token_ids
            text = pred.outputs[0].text
            logprobs = pred.outputs[0].logprobs
            seq_score = pred.outputs[0].cumulative_logprob / max(len(token_ids), 1)

            # ISREL
            rel_dict = {}
            for tok, tid in rel_tokens.items():
                prob = logprobs[0][tid] if tid in logprobs[0] else -100
                rel_dict[tok] = np.exp(prob)
            rel_sum = np.sum(list(rel_dict.values())) or 1.0
            relevance_score = rel_dict["[Relevant]"] / rel_sum

            # ISSUP
            grd_dict = {}
            if grd_tokens is not None:
                grd_vals = list(grd_tokens.values())
                for i, tid in enumerate(token_ids):
                    if tid in grd_vals:
                        for tok, tok_id in grd_tokens.items():
                            prob = logprobs[i][tok_id] if tok_id in logprobs[i] else -100
                            grd_dict[tok] = np.exp(prob)
                        break
            if len(grd_dict) == 3:
                gt_sum = np.sum(list(grd_dict.values())) or 1.0
                ground_score = (grd_dict["[Fully supported]"] / gt_sum) + 0.5 * (
                    grd_dict["[Partially supported]"] / gt_sum)
            else:
                ground_score = 0.0

            # ISUSE
            ut_dict = {}
            if ut_tokens is not None:
                ut_vals = list(ut_tokens.values())
                for i, tid in enumerate(token_ids):
                    if tid in ut_vals:
                        # Note: original code uses grd_tokens here (bug in original code)
                        for tok, tok_id in grd_tokens.items():
                            prob = logprobs[i][tok_id] if tok_id in logprobs[i] else -100
                            ut_dict[tok] = np.exp(prob)
                        break
            if len(ut_dict) == 5:
                ut_sum = np.sum(list(ut_dict.values())) or 1.0
                ut_scale = [-1, -0.5, 0, 0.5, 1]
                utility_score = np.sum([
                    ut_scale[i] * (ut_dict.get("[Utility:{}]".format(i+1), 0.0) / ut_sum)
                    for i in range(5)
                ])
            else:
                utility_score = 0.0

            if use_seqscore:
                final_score = np.exp(seq_score) + w_rel * relevance_score + \
                    w_sup * ground_score + w_use * utility_score
            else:
                final_score = w_rel * relevance_score + \
                    w_sup * ground_score + w_use * utility_score

            overall_scores[p_idx] = {
                "final_score": float(final_score),
                "relevance_score": float(relevance_score),
                "ground_score": float(ground_score),
                "utility_score": float(utility_score),
            }

            # Handle [No Retrieval] -> [Retrieval] remapping
            if "[No Retrieval]" in text:
                ret_token_appear_indices = []
                substrings = text.split("[No Retrieval]")
                for tok_idx, tok in enumerate(token_ids):
                    if tok == ret_tokens["[No Retrieval]"]:
                        ret_token_appear_indices.append(tok_idx)

                retrieval_remap = {}
                for order, idx in enumerate(ret_token_appear_indices):
                    ret_score = {}
                    for tok, tok_id in ret_tokens.items():
                        prob = logprobs[idx][tok_id] if tok_id in logprobs[idx] else -100
                        ret_score[tok] = np.exp(prob)
                    denom = ret_score["[Retrieval]"] + ret_score["[No Retrieval]"]
                    if denom != 0.0:
                        do_ret = (ret_score["[Retrieval]"] + ret_score.get("[Continue to Use Evidence]", 0.0)) / denom > threshold
                    else:
                        do_ret = False
                    retrieval_remap[order] = bool(do_ret)

                processed = ""
                for si, substring in enumerate(substrings):
                    if si in retrieval_remap and retrieval_remap[si]:
                        processed += substring + "[Retrieval]"
                    else:
                        processed += substring + "[No Retrieval]"
                text = processed

            final_preds.append(text)
            scores.append(float(final_score))

        return final_preds, scores, overall_scores

    def beam_step_node(state):
        tree = dict(state.get("prediction_tree", {}))
        levels = dict(state.get("levels", {}))
        depth = state.get("current_depth", 1)
        nid = state.get("node_id_counter", 0)
        terminated = state.get("terminated", False)
        beam_width = state.get("beam_width", getattr(config, "beam_width", 2))
        max_depth = state.get("max_depth", getattr(config, "max_depth", 7))
        ctxs = state.get("docs", [])[:state.get("ndocs", config.ndocs)]

        w_rel = state.get("w_rel", config.w_rel)
        w_sup = state.get("w_sup", config.w_sup)
        w_use = state.get("w_use", config.w_use)
        use_seq = state.get("use_seqscore", config.use_seqscore)
        threshold = state.get("threshold", config.threshold)
        max_new_tokens = state.get("max_new_tokens", config.max_new_tokens)

        levels[depth] = []

        if (depth - 1) in levels and not terminated:
            for node in levels[depth - 1]:
                pred = tree[node]["pred"]
                if pred == "</s>":
                    terminated = True
                    continue
                prompt = tree[node]["prompt"]
                prev_gen = tree[node]["processed_pred"]
                score = tree[node]["score"]

                if "[Retrieval]" in pred:
                    preds, scores, overall = _run_step_generation(
                        prompt + prev_gen, ctxs, max_new_tokens,
                        w_rel, w_sup, w_use, use_seq, threshold,
                    )
                    for i, (p, s) in enumerate(zip(preds, scores)):
                        nid += 1
                        node_score = s * score if score is not None else s
                        tree[nid] = {
                            "prompt": prompt + prev_gen,
                            "pred": p,
                            "score": float(node_score),
                            "ctx": ctxs[i] if i < len(ctxs) else None,
                            "parent": node,
                            "overall_score_dict": overall,
                        }
                        if "[Retrieval]" in p:
                            gen_idx = p.index("[Retrieval]")
                            tree[nid]["processed_pred"] = p[:gen_idx]
                        else:
                            tree[nid]["processed_pred"] = p
                        levels[depth].append(nid)

        # Prune to beam_width
        if levels[depth]:
            node2score = {n: tree[n]["score"] for n in levels[depth]}
            top = sorted(node2score.items(), key=lambda x: x[1], reverse=True)[:beam_width]
            levels[depth] = [n for n, _ in top]

        return {
            "prediction_tree": tree,
            "levels": levels,
            "current_depth": depth + 1,
            "node_id_counter": nid,
            "terminated": terminated,
        }

    return beam_step_node