"""Unified Self-RAG evaluation through WTB (Workflow Test Bench).

Supports HotPotQA, UltraDomain, and Self-RAG's original datasets (PopQA, etc.)
All execution goes through bench.run() → ExecutionController → LangGraphStateAdapter.

Prerequisites:
    pip install langgraph-checkpoint-sqlite   # WTB LangGraph adapter needs this

Usage:
    # HotPotQA
    python run_eval.py \
        --dataset hotpotqa \
        --input data/hotpotqa_val_contriever.jsonl \
        --output results/hotpotqa_wtb.jsonl \
        --model_name selfrag/selfrag_llama2_7b \
        --mode always_retrieve --ndocs 10

    # UltraDomain
    python run_eval.py \
        --dataset ultradomain \
        --input data/ultradomain_contriever.jsonl \
        --output results/ultradomain_wtb.jsonl \
        --model_name selfrag/selfrag_llama2_7b \
        --mode always_retrieve --ndocs 10

    # PopQA (original Self-RAG dataset)
    python run_eval.py \
        --dataset popqa \
        --input data/popqa_longtail.jsonl \
        --output results/popqa_wtb.jsonl \
        --model_name selfrag/selfrag_llama2_7b \
        --mode adaptive_retrieval --ndocs 5
"""

import argparse
import json
import os
import re
import string
import time
from collections import Counter
from tqdm import tqdm


# =========================================================================
# Metrics
# =========================================================================

def normalize_answer(s: str) -> str:
    """Lowercase, remove articles, punctuation, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 (SQuAD / HotPotQA standard)."""
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


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_multi_answer_f1(prediction: str, answers: list[str]) -> float:
    """Max F1 over multiple reference answers (UltraDomain)."""
    if not answers:
        return 0.0
    return max(compute_f1(prediction, a) for a in answers)


def compute_multi_answer_em(prediction: str, answers: list[str]) -> float:
    """Max EM over multiple reference answers."""
    if not answers:
        return 0.0
    return max(compute_exact_match(prediction, a) for a in answers)


# =========================================================================
# Data loading
# =========================================================================

def load_data(input_path: str, num_samples: int = 0) -> list[dict]:
    """Load JSONL data file."""
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    if num_samples > 0:
        data = data[:num_samples]
    return data


def build_initial_state(item: dict, dataset: str, ndocs: int) -> dict:
    """Build the initial state dict for WTB bench.run().

    Maps dataset-specific fields to QueryState keys.
    """
    evidences = item.get("evidences", [])[:ndocs]
    state = {
        "question": item.get("question", ""),
        "evidences": evidences,
        "dataset": dataset,
    }

    # Multi-answer support (UltraDomain)
    if "answers" in item and isinstance(item["answers"], list):
        state["answers"] = item["answers"]

    # Domain label (UltraDomain)
    if "domain" in item:
        state["domain"] = item["domain"]

    return state


# =========================================================================
# Main evaluation
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Self-RAG unified evaluation through WTB"
    )
    # Data
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["hotpotqa", "ultradomain", "popqa", "triviaqa",
                                 "arc_c", "fever", "asqa"],
                        help="Dataset name")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file (from prepare_*.py)")
    parser.add_argument("--output", type=str, default="results/eval_results.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Number of samples to evaluate (0 = all)")

    # Model
    parser.add_argument("--model_name", type=str, default="selfrag/selfrag_llama2_7b")
    parser.add_argument("--download_dir", type=str,
                        default="/data1/ragworkspace/self-rag/model_cache")
    parser.add_argument("--dtype", type=str, default="half")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)

    # Self-RAG config
    parser.add_argument("--mode", type=str, default="always_retrieve",
                        choices=["always_retrieve", "adaptive_retrieval", "no_retrieval"])
    parser.add_argument("--ndocs", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--w_rel", type=float, default=1.0)
    parser.add_argument("--w_sup", type=float, default=1.0)
    parser.add_argument("--w_use", type=float, default=0.5)
    parser.add_argument("--use_seqscore", action="store_true")

    # WTB config
    parser.add_argument("--wtb_mode", type=str, default="development",
                        choices=["testing", "development"],
                        help="WTB bench mode (development uses SQLite persistence)")
    parser.add_argument("--wtb_data_dir", type=str, default="wtb_data",
                        help="WTB data directory (for development mode)")
    args = parser.parse_args()

    # ---- Load evaluation data ----
    print(f"[INFO] Loading {args.dataset} data from {args.input}")
    data = load_data(args.input, args.num_samples)
    print(f"[INFO] Loaded {len(data)} examples")

    # ---- Load Self-RAG model (vLLM) ----
    print(f"[INFO] Loading Self-RAG model: {args.model_name}")
    from vllm import LLM
    from transformers import AutoTokenizer

    llm_tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, cache_dir=args.download_dir,
    )
    llm_model = LLM(
        model=args.model_name,
        download_dir=args.download_dir,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_logprobs=50000,
    )
    print("[INFO] Self-RAG model loaded")

    # ---- Build Self-RAG config ----
    from selfrag.config import SelfRAGConfig
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

    # ---- Create WTB project and bench ----
    print("[INFO] Creating WTB bench and registering Self-RAG project...")
    from wtb.sdk.test_bench import WTBTestBench
    from wtb_integration import create_selfrag_query_project

    bench = WTBTestBench.create(
        mode=args.wtb_mode,
        data_dir=args.wtb_data_dir,
    )
    project = create_selfrag_query_project(
        llm_model=llm_model,
        tokenizer=llm_tokenizer,
        config=config,
    )
    bench.register_project(project)
    print(f"[INFO] WTB bench ready (mode={args.wtb_mode}), project='{project.name}'")

    # ---- Run evaluation ----
    os.makedirs(
        os.path.dirname(args.output) if os.path.dirname(args.output) else ".",
        exist_ok=True,
    )

    all_f1 = []
    all_em = []
    latencies = []
    domain_metrics = {}  # For UltraDomain per-domain breakdown

    fout = open(args.output, "w", encoding="utf-8")

    for idx, item in enumerate(tqdm(data, desc=f"Evaluating {args.dataset}")):
        initial_state = build_initial_state(item, args.dataset, args.ndocs)

        t0 = time.time()
        try:
            execution = bench.run(
                project=project.name,
                initial_state=initial_state,
            )
            # Extract final_pred from WTB execution state
            final_state = execution.state.workflow_variables
            pred = final_state.get("final_pred", "")
        except Exception as e:
            print(f"\n[ERROR] Example {idx} ({item.get('id', '?')}): {e}")
            pred = ""
        elapsed = time.time() - t0
        latencies.append(elapsed)

        # Compute metrics based on dataset type
        answers = item.get("answers")
        answer = item.get("answer", "")

        if args.dataset == "ultradomain" and answers and isinstance(answers, list):
            f1 = compute_multi_answer_f1(pred, answers)
            em = compute_multi_answer_em(pred, answers)
        else:
            f1 = compute_f1(pred, answer)
            em = compute_exact_match(pred, answer)

        all_f1.append(f1)
        all_em.append(em)

        # Per-domain tracking (UltraDomain)
        domain = item.get("domain")
        if domain:
            if domain not in domain_metrics:
                domain_metrics[domain] = {"f1": [], "em": []}
            domain_metrics[domain]["f1"].append(f1)
            domain_metrics[domain]["em"].append(em)

        record = {
            "id": item.get("id", f"{args.dataset}-{idx}"),
            "question": item.get("question", ""),
            "answer": answer,
            "prediction": pred,
            "f1": round(f1, 4),
            "em": round(em, 4),
            "latency_s": round(elapsed, 3),
        }
        if domain:
            record["domain"] = domain
        if answers:
            record["answers"] = answers

        fout.write(json.dumps(record, ensure_ascii=False) + "\n")

        if (idx + 1) % 50 == 0:
            avg_f1 = sum(all_f1) / len(all_f1) * 100
            avg_em = sum(all_em) / len(all_em) * 100
            print(f"\n  [Progress {idx+1}/{len(data)}] "
                  f"F1={avg_f1:.2f}%  EM={avg_em:.2f}%  "
                  f"Latency={sum(latencies)/len(latencies):.2f}s")

    fout.close()
    bench.close()

    # ---- Final metrics ----
    avg_f1 = sum(all_f1) / len(all_f1) * 100 if all_f1 else 0.0
    avg_em = sum(all_em) / len(all_em) * 100 if all_em else 0.0

    print("\n" + "=" * 60)
    print(f"Self-RAG {args.dataset.upper()} Evaluation Results (via WTB)")
    print("=" * 60)
    print(f"  Dataset:        {args.dataset}")
    print(f"  Mode:           {args.mode}")
    print(f"  Num examples:   {len(all_f1)}")
    print(f"  Ndocs:          {args.ndocs}")
    print(f"  F1:             {avg_f1:.2f}%")
    print(f"  Exact Match:    {avg_em:.2f}%")
    print(f"  Avg Latency:    {sum(latencies)/len(latencies):.2f}s")
    print("=" * 60)

    # Per-domain breakdown (UltraDomain)
    if domain_metrics:
        print("\n  Per-domain breakdown:")
        for d in sorted(domain_metrics.keys()):
            dm = domain_metrics[d]
            d_f1 = sum(dm["f1"]) / len(dm["f1"]) * 100
            d_em = sum(dm["em"]) / len(dm["em"]) * 100
            print(f"    {d:20s}  F1={d_f1:6.2f}%  EM={d_em:6.2f}%  (n={len(dm['f1'])})")

    print(f"\n  Predictions saved to: {args.output}")

    # ---- Save summary ----
    summary_path = args.output.replace(".jsonl", "_summary.json")
    summary = {
        "dataset": args.dataset,
        "mode": args.mode,
        "model": args.model_name,
        "num_examples": len(all_f1),
        "ndocs": args.ndocs,
        "max_new_tokens": args.max_new_tokens,
        "f1": round(avg_f1, 4),
        "exact_match": round(avg_em, 4),
        "avg_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else 0,
        "wtb_mode": args.wtb_mode,
    }
    if domain_metrics:
        summary["per_domain"] = {
            d: {
                "f1": round(sum(dm["f1"]) / len(dm["f1"]) * 100, 4),
                "em": round(sum(dm["em"]) / len(dm["em"]) * 100, 4),
                "count": len(dm["f1"]),
            }
            for d, dm in sorted(domain_metrics.items())
        }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
