"""Compare the monolithic long-form Self-RAG pipeline with the LangGraph
long-form pipeline on ASQA (ALCE benchmark).

Both pipelines share the same vLLM model, same reflection-token maps, same
pre-retrieved ctxs. Any gap should be ~0 if the refactor is faithful.

Usage:
    python run_compare_eval_longform.py \
        --model selfrag/selfrag_llama2_7b \
        --download_dir /data1/ragworkspace/self-rag/model_cache \
        --output_dir /data1/ragworkspace/self-rag/eval_results \
        --input_file /data1/ragworkspace/self-rag/ALCE-main/data/asqa_eval_gtr_top100.json \
        --task asqa --ndocs 5 --max_new_tokens 300 --beam_width 2 --max_depth 7
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM

from run_long_form_static import call_model_beam_batch
from utils import PROMPT_DICT, TASK_INST, load_special_tokens, postprocess, fix_spacing

from langgraph_nodes import build_longform_query_graph, SelfRAGConfig


def run_original_pipeline(
    model, tokenizer, data, args,
    ret_tokens, rel_tokens, grd_tokens, ut_tokens,
):
    """Run the original long-form pipeline from run_long_form_static.py."""
    results = {"data": [], "args": [], "total_cost": 0.0, "azure_filter_fail": ""}

    def generate(prompt, ctxs, max_new_tokens):
        processed = PROMPT_DICT["prompt_no_input"].format_map({"instruction": prompt})
        return call_model_beam_batch(
            processed, model=model, max_new_tokens=max_new_tokens,
            ctxs=ctxs, query=prompt,
            rel_tokens=rel_tokens, ret_tokens=ret_tokens,
            grd_tokens=grd_tokens, ut_tokens=ut_tokens,
            use_seqscore=args.use_seqscore, threshold=args.threshold,
            beam_width=args.beam_width, max_depth=args.max_depth,
            w_rel=args.w_rel, w_sup=args.w_sup, w_use=args.w_use,
            mode=args.mode, ignore_cont=args.ignore_cont,
        )

    t0 = time.time()
    for idx, item in enumerate(tqdm(data, desc="original")):
        prompt = item["question"]
        ctxs = item["docs"][:args.ndocs]
        instructions = TASK_INST[args.task]
        prompt = instructions + "## Input:\n\n" + prompt

        final_pred, intermediate = generate(prompt, ctxs, args.max_new_tokens)

        # Post-process (same as run_long_form_static.py main())
        final_output = ""
        docs = []
        prev_gen = []
        if "splitted_sentences" not in intermediate:
            item["output"] = postprocess(final_pred)
        else:
            if len(postprocess(final_pred[0])) == 0:
                intermediate["splitted_sentences"][0] = intermediate["splitted_sentences"][1]
                intermediate["ctxs"][0] = intermediate["ctxs"][1]
            for si, (sent, doc) in enumerate(zip(
                intermediate["splitted_sentences"][0],
                intermediate["ctxs"][0]
            )):
                if len(sent) == 0:
                    continue
                pp = postprocess(sent)
                if pp in prev_gen:
                    continue
                prev_gen.append(pp)
                final_output += pp[:-1] + " [{}]".format(si) + ". "
                docs.append(doc)
            if final_output and final_output[-1] == " ":
                final_output = final_output[:-1]
            final_output = fix_spacing(final_output)
            final_output = final_output.replace(".[Continue to Use Evidence]", " [1]. ")
            final_output = final_output.replace(". [1] ", " [1]. ")
            item["output"] = final_output

        item["docs"] = docs
        if "original_splitted_sentences" in intermediate:
            item["intermediate"] = intermediate["original_splitted_sentences"][0]
        results["data"].append(item)

        if idx % 10 == 0:
            print(f"  [{idx}/{len(data)}] output: {item['output'][:80]}...")

    dt = time.time() - t0
    return results, dt


def run_langgraph_pipeline(graph, data, args):
    """Run the LangGraph long-form pipeline."""
    results = {"data": [], "args": [], "total_cost": 0.0, "azure_filter_fail": ""}

    t0 = time.time()
    for idx, item in enumerate(tqdm(data, desc="langgraph")):
        state = {
            "question": item["question"],
            "task": args.task,
            "docs": item["docs"][:args.ndocs],
            "mode": args.mode,
            "threshold": args.threshold,
            "w_rel": args.w_rel,
            "w_sup": args.w_sup,
            "w_use": args.w_use,
            "use_seqscore": args.use_seqscore,
            "max_new_tokens": args.max_new_tokens,
            "ndocs": args.ndocs,
            "beam_width": args.beam_width,
            "max_depth": args.max_depth,
            "ignore_cont": args.ignore_cont,
        }
        out = graph.invoke(state)
        item_copy = dict(item)
        item_copy["output"] = out.get("final_output", "")
        item_copy["docs"] = out.get("output_docs", [])
        if out.get("intermediate"):
            inter = out["intermediate"]
            if "original_splitted_sentences" in inter and 0 in inter["original_splitted_sentences"]:
                item_copy["intermediate"] = inter["original_splitted_sentences"][0]
        results["data"].append(item_copy)

        if idx % 10 == 0:
            print(f"  [{idx}/{len(data)}] output: {item_copy['output'][:80]}...")

    dt = time.time() - t0
    return results, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="selfrag/selfrag_llama2_7b")
    ap.add_argument("--download_dir", default="/data1/ragworkspace/self-rag/model_cache")
    ap.add_argument("--output_dir", default="/data1/ragworkspace/self-rag/eval_results")
    ap.add_argument("--input_file", default="/data1/ragworkspace/self-rag/ALCE-main/data/asqa_eval_gtr_top100.json")
    ap.add_argument("--task", default="asqa")
    ap.add_argument("--ndocs", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=300)
    ap.add_argument("--beam_width", type=int, default=2)
    ap.add_argument("--max_depth", type=int, default=7)
    ap.add_argument("--mode", default="adaptive_retrieval",
                    choices=["adaptive_retrieval", "always_retrieve", "no_retrieval"])
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--use_seqscore", action="store_true")
    ap.add_argument("--w_rel", type=float, default=1.0)
    ap.add_argument("--w_sup", type=float, default=1.0)
    ap.add_argument("--w_use", type=float, default=0.5)
    ap.add_argument("--ignore_cont", action="store_true")
    ap.add_argument("--dtype", default="half")
    ap.add_argument("--num_samples", type=int, default=-1)
    ap.add_argument("--skip_original", action="store_true",
                    help="Skip the original pipeline (use existing output file).")
    ap.add_argument("--original_output", type=str, default=None,
                    help="Path to existing original pipeline output.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load input data
    print(f"Loading {args.input_file} ...")
    with open(args.input_file) as f:
        input_data = json.load(f)
    if args.num_samples > 0:
        input_data = input_data[:args.num_samples]
    print(f"Loaded {len(input_data)} samples.")

    # Load model
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

    # ---- Original pipeline ----
    if args.skip_original and args.original_output:
        print(f"\nLoading existing original output from {args.original_output}")
        with open(args.original_output) as f:
            orig_results = json.load(f)
        orig_dt = 0.0
    else:
        print("\n========== Original (monolithic) pipeline ==========")
        orig_results, orig_dt = run_original_pipeline(
            model, tokenizer, input_data, args,
            ret_tokens, rel_tokens, grd_tokens, ut_tokens,
        )
        orig_path = os.path.join(args.output_dir, f"orig_{args.task}_output.json")
        with open(orig_path, "w") as f:
            json.dump(orig_results, f, ensure_ascii=False)
        print(f"Original output saved to {orig_path}  time={orig_dt:.1f}s")

    # ---- LangGraph pipeline ----
    print("\n========== LangGraph (long-form) pipeline ==========")
    cfg = SelfRAGConfig(
        model_name=args.model,
        download_dir=args.download_dir,
        dtype=args.dtype,
        ndocs=args.ndocs,
        mode=args.mode,
        threshold=args.threshold,
        max_new_tokens=args.max_new_tokens,
        beam_width=args.beam_width,
        max_depth=args.max_depth,
        ignore_cont=args.ignore_cont,
        w_rel=args.w_rel,
        w_sup=args.w_sup,
        w_use=args.w_use,
        use_seqscore=args.use_seqscore,
    )
    graph = build_longform_query_graph(
        model, ret_tokens, rel_tokens, grd_tokens, ut_tokens, config=cfg,
    )

    # Re-load input data (original pipeline may have mutated items)
    with open(args.input_file) as f:
        input_data_lg = json.load(f)
    if args.num_samples > 0:
        input_data_lg = input_data_lg[:args.num_samples]

    lg_results, lg_dt = run_langgraph_pipeline(graph, input_data_lg, args)
    lg_path = os.path.join(args.output_dir, f"lg_{args.task}_output.json")
    with open(lg_path, "w") as f:
        json.dump(lg_results, f, ensure_ascii=False)
    print(f"LangGraph output saved to {lg_path}  time={lg_dt:.1f}s")

    # ---- Compare outputs ----
    print("\n========== Comparison ==========")
    orig_items = orig_results["data"]
    lg_items = lg_results["data"]
    n = min(len(orig_items), len(lg_items))
    exact_match = sum(
        1 for i in range(n)
        if orig_items[i].get("output", "") == lg_items[i].get("output", "")
    )
    print(f"Exact output agreement: {exact_match}/{n} ({100.0*exact_match/max(n,1):.2f}%)")
    print(f"Original time: {orig_dt:.1f}s, LangGraph time: {lg_dt:.1f}s")
    print(f"\nTo run ALCE evaluation:")
    print(f"  python /data1/ragworkspace/self-rag/ALCE-main/eval.py --f {os.path.join(args.output_dir, 'orig_' + args.task + '_output.json')} --qa --no_rouge")
    print(f"  python /data1/ragworkspace/self-rag/ALCE-main/eval.py --f {lg_path} --qa --no_rouge")


if __name__ == "__main__":
    main()
