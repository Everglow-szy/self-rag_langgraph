"""Compare the monolithic self-RAG pipeline (run_short_form.py) with the
langgraph-node pipeline on PopQA.

Both pipelines share the same vLLM model, same reflection-token maps, same
pre-retrieved ctxs.  Any gap should be ~0 if the refactor is faithful.

Usage:
    python run_compare_eval.py --num_samples 500 \
        --model selfrag/selfrag_llama2_7b \
        --download_dir /data1/ragworkspace/self-rag/model_cache \
        --output_dir /data1/ragworkspace/self-rag/eval_results
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM

# self-rag monolithic code (left untouched)
from run_short_form import call_model_rerank_w_scores_batch, postprocess_answer_option_conditioned
from utils import PROMPT_DICT, load_special_tokens
from metrics import match as metric_match

from langgraph_nodes import build_query_graph, SelfRAGConfig


SEED = 633


def load_popqa_with_ctxs(num_samples: int, seed: int = SEED):
    """Load the PopQA long-tail subset (s_pop < 100, 1399 examples) with the
    same Contriever-Wikipedia pre-retrieved ctxs used by the Self-RAG paper.

    Per the paper (section 4.1): *"For PopQA, we use the long-tail subset,
    consisting of 1,399 rare entity queries whose monthly Wikipedia page views
    are less than 100"* and *"We evaluate performance based on whether gold
    answers are included in the model generations"* (i.e. substring match).

    We take questions + gold answers from ``akariasai/PopQA`` (the dataset the
    user asked us to use) and attach ``ctxs`` from ``awinml/popqa_longtail_w_gs``
    which is a HuggingFace mirror of Self-RAG's official
    ``eval_data/popqa_longtail_w_gs.jsonl`` (same 1,399 ids, same Contriever
    passages).  Matching is done on the shared ``id`` field.
    """
    from datasets import load_dataset

    print("Loading akariasai/PopQA ...")
    popqa = load_dataset("akariasai/PopQA", split="test")
    print("Loading awinml/popqa_longtail_w_gs (Self-RAG's pre-retrieved ctxs) ...")
    pre = load_dataset("awinml/popqa_longtail_w_gs", split="train")

    id2ctxs = {str(row["id"]): row["ctxs"] for row in pre}

    matched = []
    for row in popqa:
        # Long-tail filter: monthly Wikipedia page views < 100
        if int(row["s_pop"]) >= 100:
            continue
        rid = str(row["id"])
        if rid not in id2ctxs:
            continue
        try:
            answers = json.loads(row["possible_answers"])
        except Exception:
            answers = [row["possible_answers"]]
        matched.append(
            {
                "id": rid,
                "question": row["question"],
                "answers": answers,
                "instruction": row["question"],
                "ctxs": id2ctxs[rid],
            }
        )
    # Deterministic order by id so both pipelines see the exact same sequence
    matched.sort(key=lambda x: int(x["id"]))
    if num_samples > 0:
        matched = matched[:num_samples]
    print(f"Prepared {len(matched)} PopQA long-tail examples with ctxs.")
    return matched


def run_original_pipeline(
    model, tokenizer, data, ret_tokens, rel_tokens, grd_tokens, ut_tokens,
    ndocs: int, max_new_tokens: int, mode: str,
    use_seqscore: bool, w_rel: float, w_sup: float, w_use: float,
):
    preds = []
    retrieval_used = 0
    t0 = time.time()
    for row in tqdm(data, desc="original"):
        prompt = PROMPT_DICT["prompt_no_input"].format_map(row)
        evidences = row["ctxs"][:ndocs]
        pred, _results, do_retrieve = call_model_rerank_w_scores_batch(
            prompt,
            evidences=evidences,
            model=model,
            max_new_tokens=max_new_tokens,
            ret_tokens=ret_tokens,
            rel_tokens=rel_tokens,
            grd_tokens=grd_tokens,
            ut_tokens=ut_tokens,
            threshold=0.2,
            use_seqscore=use_seqscore,
            w_rel=w_rel,
            w_sup=w_sup,
            w_use=w_use,
            mode=mode,
            closed=False,
        )
        if isinstance(pred, str) and pred and pred[0] in ("#", ":"):
            pred = pred[1:]
        preds.append(pred)
        if do_retrieve:
            retrieval_used += 1
    dt = time.time() - t0
    return preds, retrieval_used, dt


def run_langgraph_pipeline(
    graph, data, ndocs: int, max_new_tokens: int, mode: str,
    use_seqscore: bool, w_rel: float, w_sup: float, w_use: float,
):
    preds = []
    retrieval_used = 0
    t0 = time.time()
    for row in tqdm(data, desc="langgraph"):
        state = {
            "question": row["instruction"],
            "gold_answers": row["answers"],
            "evidences": row["ctxs"][:ndocs],
            "mode": mode,
            "threshold": 0.2,
            "w_rel": w_rel,
            "w_sup": w_sup,
            "w_use": w_use,
            "use_seqscore": use_seqscore,
            "max_new_tokens": max_new_tokens,
            "closed": False,
            "ndocs": ndocs,
        }
        out = graph.invoke(state)
        preds.append(out.get("final_pred", ""))
        if out.get("do_retrieve"):
            retrieval_used += 1
    dt = time.time() - t0
    return preds, retrieval_used, dt


def score_match(preds, data):
    return 100.0 * float(np.mean([metric_match(p, row["answers"]) for p, row in zip(preds, data)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="selfrag/selfrag_llama2_7b")
    ap.add_argument("--download_dir", default="/data1/ragworkspace/self-rag/model_cache")
    ap.add_argument("--output_dir", default="/data1/ragworkspace/self-rag/eval_results")
    ap.add_argument("--num_samples", type=int, default=-1,
                    help="-1 to run the full 1,399-example long-tail subset.")
    ap.add_argument("--ndocs", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=100,
                    help="Paper uses 100 tokens for short-form open QA (Appx B.2).")
    ap.add_argument("--mode", default="adaptive_retrieval",
                    choices=["adaptive_retrieval", "always_retrieve", "no_retrieval"])
    ap.add_argument("--use_seqscore", action="store_true")
    ap.add_argument("--w_rel", type=float, default=1.0)
    ap.add_argument("--w_sup", type=float, default=1.0)
    ap.add_argument("--w_use", type=float, default=0.5)
    ap.add_argument("--dtype", default="half")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # ------------------------------------------------------------------
    # Data (user-requested: load via datasets)
    # ------------------------------------------------------------------
    data = load_popqa_with_ctxs(args.num_samples)

    # ------------------------------------------------------------------
    # Model (loaded once, shared by both pipelines)
    # ------------------------------------------------------------------
    print(f"Loading LLM {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = LLM(
        model=args.model,
        download_dir=args.download_dir,
        dtype=args.dtype,
        tensor_parallel_size=1,
    )
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=True, use_utility=True
    )
    print("loaded reflection tokens:",
          {k: len(v) if v else 0 for k, v in
           [("ret", ret_tokens), ("rel", rel_tokens),
            ("grd", grd_tokens), ("ut", ut_tokens)]})

    # ------------------------------------------------------------------
    # Run original pipeline
    # ------------------------------------------------------------------
    print("\n========== Original (monolithic) pipeline ==========")
    orig_preds, orig_retr, orig_dt = run_original_pipeline(
        model, tokenizer, data, ret_tokens, rel_tokens, grd_tokens, ut_tokens,
        args.ndocs, args.max_new_tokens, args.mode,
        args.use_seqscore, args.w_rel, args.w_sup, args.w_use,
    )
    orig_score = score_match(orig_preds, data)
    print(f"Original match={orig_score:.2f}  retr={orig_retr}/{len(data)}  time={orig_dt:.1f}s")

    # ------------------------------------------------------------------
    # Run langgraph pipeline
    # ------------------------------------------------------------------
    print("\n========== LangGraph (pluggable-node) pipeline ==========")
    cfg = SelfRAGConfig(
        model_name=args.model,
        download_dir=args.download_dir,
        dtype=args.dtype,
        ndocs=args.ndocs,
        mode=args.mode,
        threshold=0.2,
        max_new_tokens=args.max_new_tokens,
        w_rel=args.w_rel,
        w_sup=args.w_sup,
        w_use=args.w_use,
        use_seqscore=args.use_seqscore,
        closed=False,
    )
    graph = build_query_graph(
        model, ret_tokens, rel_tokens, grd_tokens, ut_tokens,
        config=cfg,
    )
    lg_preds, lg_retr, lg_dt = run_langgraph_pipeline(
        graph, data, args.ndocs, args.max_new_tokens, args.mode,
        args.use_seqscore, args.w_rel, args.w_sup, args.w_use,
    )
    lg_score = score_match(lg_preds, data)
    print(f"LangGraph match={lg_score:.2f}  retr={lg_retr}/{len(data)}  time={lg_dt:.1f}s")

    # ------------------------------------------------------------------
    # Agreement between the two pipelines
    # ------------------------------------------------------------------
    same = sum(1 for a, b in zip(orig_preds, lg_preds) if a == b)
    agreement = 100.0 * same / max(len(data), 1)
    abs_gap = abs(orig_score - lg_score)
    print("\n========== Comparison ==========")
    print(f"Exact prediction agreement: {same}/{len(data)} ({agreement:.2f}%)")
    print(f"Match accuracy delta: {abs_gap:.2f} pts")
    significant = abs_gap > 1.0
    print(f"Significant gap? {'YES – investigate!' if significant else 'no (<=1 pt)'}")

    # ------------------------------------------------------------------
    # Dump everything for inspection
    # ------------------------------------------------------------------
    out = {
        "config": vars(args),
        "num_samples": len(data),
        "original": {
            "match": orig_score,
            "retrieval_calls": orig_retr,
            "runtime_sec": orig_dt,
        },
        "langgraph": {
            "match": lg_score,
            "retrieval_calls": lg_retr,
            "runtime_sec": lg_dt,
        },
        "comparison": {
            "exact_agreement_pct": agreement,
            "abs_match_gap": abs_gap,
            "significant": bool(significant),
        },
        "per_sample": [
            {
                "question": row["question"],
                "gold": row["answers"],
                "orig_pred": op,
                "lg_pred": lp,
                "match_orig": metric_match(op, row["answers"]),
                "match_lg": metric_match(lp, row["answers"]),
            }
            for row, op, lp in zip(data, orig_preds, lg_preds)
        ],
    }
    out_path = os.path.join(args.output_dir, "compare_popqa.json")
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
