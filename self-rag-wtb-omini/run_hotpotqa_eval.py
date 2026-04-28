"""HotPotQA evaluation with Self-RAG via WTB project structure.

Loads pre-processed HotPotQA data (from prepare_hotpotqa.py), builds the
Self-RAG query graph via wtb_integration, compiles with LangGraph checkpointer,
and evaluates with token-level F1.

Usage:
    python run_hotpotqa_eval.py \
        --input data/hotpotqa_val_contriever.jsonl \
        --output results/hotpotqa_results.jsonl \
        --model_name selfrag/selfrag_llama2_7b \
        --download_dir /data1/ragworkspace/self-rag/model_cache \
        --num_samples 500 \
        --mode always_retrieve \
        --ndocs 10
"""

import argparse
import json
import os
import re
import string
import sys
import time
from collections import Counter
from tqdm import tqdm


# =========================================================================
# F1 metric (standard SQuAD / HotPotQA token-level F1)
# =========================================================================

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not gt_tokens:
        return float(not pred_tokens)
    if not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


# =========================================================================
# Main evaluation
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Self-RAG on HotPotQA")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/hotpotqa_results.jsonl")
    parser.add_argument("--model_name", type=str, default="selfrag/selfrag_llama2_7b")
    parser.add_argument("--download_dir", type=str,
                        default="/data1/ragworkspace/self-rag/model_cache")
    parser.add_argument("--dtype", type=str, default="half")
    parser.add_argument("--mode", type=str, default="always_retrieve",
                        choices=["always_retrieve", "adaptive_retrieval", "no_retrieval"])
    parser.add_argument("--ndocs", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--w_rel", type=float, default=1.0)
    parser.add_argument("--w_sup", type=float, default=1.0)
    parser.add_argument("--w_use", type=float, default=0.5)
    parser.add_argument("--use_seqscore", action="store_true")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = parser.parse_args()

    # ---- Load data ----
    print(f"[INFO] Loading data from {args.input}")
    data = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    if args.num_samples > 0:
        data = data[:args.num_samples]
    print(f"[INFO] Loaded {len(data)} examples")

    # ---- Load Self-RAG model (vLLM) ----
    print(f"[INFO] Loading Self-RAG model: {args.model_name}")
    from vllm import LLM
    from transformers import AutoTokenizer

    llm_tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, cache_dir=args.download_dir
    )
    llm_model = LLM(
        model=args.model_name,
        download_dir=args.download_dir,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_logprobs=50000,
    )
    print("[INFO] Self-RAG model loaded")

    # ---- Build config and graph via WTB project structure ----
    from selfrag.config import SelfRAGConfig
    from selfrag.constants import load_special_tokens
    from selfrag.graph_query import build_query_graph
    from langgraph.checkpoint.memory import MemorySaver

    config = SelfRAGConfig(
        model_name=args.model_name,
        download_dir=args.download_dir,
        dtype=args.dtype,
        mode=args.mode,
        ndocs=args.ndocs,
        max_new_tokens=args.max_new_tokens,
        threshold=args.threshold,
        w_rel=args.w_rel,
        w_sup=args.w_sup,
        w_use=args.w_use,
        use_seqscore=args.use_seqscore,
    )

    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        llm_tokenizer, use_grounding=True, use_utility=True
    )

    # Build and compile graph (same structure as WTB project's graph_factory,
    # but compiled with LangGraph checkpointer for proper execution)
    print("[INFO] Building Self-RAG query graph via WTB project structure...")
    graph = build_query_graph(
        llm_model, ret_tokens, rel_tokens, grd_tokens, ut_tokens,
        config, checkpointer=MemorySaver(),
    )
    print("[INFO] Query graph compiled")

    # ---- Run evaluation ----
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    all_f1 = []
    all_em = []
    latencies = []

    fout = open(args.output, "w", encoding="utf-8")

    for idx, item in enumerate(tqdm(data, desc="Evaluating")):
        question = item["question"]
        answer = item["answer"]
        evidences = item["evidences"][:args.ndocs]

        state = {
            "question": question,
            "evidences": evidences,
        }

        t0 = time.time()
        try:
            result = graph.invoke(
                state,
                config={"configurable": {"thread_id": f"hotpotqa-{idx}"}},
            )
            pred = result.get("final_pred", "")
        except Exception as e:
            print(f"\n[ERROR] Example {idx} ({item['id']}): {e}")
            pred = ""
        elapsed = time.time() - t0
        latencies.append(elapsed)

        f1 = compute_f1(pred, answer)
        em = compute_exact_match(pred, answer)
        all_f1.append(f1)
        all_em.append(em)

        record = {
            "id": item["id"],
            "question": question,
            "answer": answer,
            "prediction": pred,
            "f1": f1,
            "em": em,
            "latency_s": round(elapsed, 3),
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")

        if (idx + 1) % 50 == 0:
            avg_f1 = sum(all_f1) / len(all_f1) * 100
            avg_em = sum(all_em) / len(all_em) * 100
            print(f"\n  [Progress {idx+1}/{len(data)}] "
                  f"F1={avg_f1:.2f}%  EM={avg_em:.2f}%  "
                  f"Latency={sum(latencies)/len(latencies):.2f}s")

    fout.close()

    # ---- Final metrics ----
    avg_f1 = sum(all_f1) / len(all_f1) * 100 if all_f1 else 0.0
    avg_em = sum(all_em) / len(all_em) * 100 if all_em else 0.0

    print("\n" + "=" * 60)
    print("HotPotQA Self-RAG Evaluation Results")
    print("=" * 60)
    print(f"  Mode:           {args.mode}")
    print(f"  Num examples:   {len(all_f1)}")
    print(f"  Ndocs:          {args.ndocs}")
    print(f"  F1:             {avg_f1:.2f}%")
    print(f"  Exact Match:    {avg_em:.2f}%")
    print(f"  Avg Latency:    {sum(latencies)/len(latencies):.2f}s")
    print("=" * 60)
    print(f"  Predictions saved to: {args.output}")

    summary_path = args.output.replace(".jsonl", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "mode": args.mode,
            "model": args.model_name,
            "num_examples": len(all_f1),
            "ndocs": args.ndocs,
            "max_new_tokens": args.max_new_tokens,
            "f1": round(avg_f1, 4),
            "exact_match": round(avg_em, 4),
            "avg_latency_s": round(sum(latencies)/len(latencies), 3) if latencies else 0,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
