"""Evaluate Self-RAG on HotPotQA / UltraDomain via OminiRAG benchmark adapters.

Bridges self-rag-wtb-eval with OminiRAG's benchmark adapters through the
rag_contracts protocol layer.  No modifications to OminiRAG or WTB code needed.

Two evaluation modes:
  1. Pipeline mode (--eval_mode pipeline):
     Builds a full LangGraph modular pipeline, passes it to
     OminiRAG's evaluate_pipeline(data, graph).
  2. Generation mode (--eval_mode generation):
     Uses SelfRAGGeneration adapter directly, passes it to
     OminiRAG's evaluate_generation(data, generation).

Usage:
    # HotPotQA - pipeline mode
    python run_eval_omini.py \
        --dataset hotpotqa \
        --input data/hotpotqa_val_contriever.jsonl \
        --output results/hotpotqa_omini.json \
        --model_name selfrag/selfrag_llama2_7b \
        --eval_mode pipeline

    # HotPotQA - generation mode (skip retrieval, use pre-retrieved evidences)
    python run_eval_omini.py \
        --dataset hotpotqa \
        --input data/hotpotqa_val_contriever.jsonl \
        --output results/hotpotqa_omini_gen.json \
        --model_name selfrag/selfrag_llama2_7b \
        --eval_mode generation

    # UltraDomain - with LLM judge
    python run_eval_omini.py \
        --dataset ultradomain \
        --input data/ultradomain_contriever.jsonl \
        --output results/ultradomain_omini.json \
        --model_name selfrag/selfrag_llama2_7b \
        --eval_mode generation \
        --judge_model gpt-4o-mini

    # UltraDomain - KG sample data from OminiRAG
    python run_eval_omini.py \
        --dataset ultradomain \
        --sample_dir /path/to/OminiRAG/benchmark/sample_data/ultradomain_kg_sample \
        --output results/ultradomain_omini.json \
        --model_name selfrag/selfrag_llama2_7b \
        --eval_mode generation
"""

import argparse
import json
import os
import sys

from rag_contracts import RetrievalResult


# =========================================================================
# Pre-retrieved evidence adapter
# =========================================================================

class PreRetrievedRetrieval:
    """A rag_contracts.Retrieval that serves pre-retrieved evidences.

    For datasets with pre-computed retrieval results (from prepare_hotpotqa.py
    or prepare_ultradomain.py), this adapter converts the evidence dicts into
    canonical RetrievalResult objects.

    Evidences are loaded per-query via set_evidences() before each call.
    """

    def __init__(self):
        self._evidences: list[dict] = []

    def set_evidences(self, evidences: list[dict]):
        self._evidences = evidences

    def retrieve(self, queries: list[str], top_k: int = 10) -> list[RetrievalResult]:
        results = []
        for i, ev in enumerate(self._evidences[:top_k]):
            results.append(RetrievalResult(
                source_id=f"evidence_{i}",
                content=ev.get("text", ""),
                score=ev.get("score", 0.0),
                title=ev.get("title", ""),
                metadata={"rank": ev.get("rank", i)},
            ))
        return results


# =========================================================================
# Data loading (unified format from prepare_*.py)
# =========================================================================

def load_jsonl(path: str, num_samples: int = 0) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    if num_samples > 0:
        data = data[:num_samples]
    return data


def jsonl_to_omini_hotpotqa(items: list[dict]) -> list[dict]:
    """Convert prepare_hotpotqa.py output to OminiRAG HotpotQABenchmarkAdapter format."""
    converted = []
    for item in items:
        chunks = {}
        for i, ev in enumerate(item.get("evidences", [])):
            chunk_id = f"chunk_{i}"
            chunks[chunk_id] = {
                "content": ev.get("text", ""),
                "doc_ids": [ev.get("title", f"doc_{i}")],
            }
        converted.append({
            "question": item["question"],
            "answer": item.get("answer", ""),
            "query_id": item.get("id", ""),
            "chunks": chunks,
        })
    return converted


def jsonl_to_omini_ultradomain(items: list[dict]) -> list[dict]:
    """Convert prepare_ultradomain.py output to OminiRAG UltraDomainBenchmarkAdapter format."""
    converted = []
    for item in items:
        chunks = {}
        for i, ev in enumerate(item.get("evidences", [])):
            chunk_id = f"chunk_{i}"
            chunks[chunk_id] = {
                "content": ev.get("text", ""),
                "doc_ids": [ev.get("title", f"doc_{i}")],
            }
        converted.append({
            "question": item["question"],
            "answer": item.get("answer", ""),
            "domain": item.get("domain", "general"),
            "query_id": item.get("id", ""),
            "chunks": chunks,
        })
    return converted


# =========================================================================
# LLM judge factory (for UltraDomain)
# =========================================================================

def create_openai_judge(model_name: str):
    """Create an OpenAI-compatible LLM judge callable for UltraDomain evaluation."""
    from openai import OpenAI
    client = OpenAI()

    def llm_complete(system_prompt: str, user_prompt: str, **kwargs) -> str:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 100),
        )
        return response.choices[0].message.content

    return llm_complete


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Self-RAG on HotPotQA/UltraDomain via OminiRAG adapters"
    )

    # Data
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["hotpotqa", "ultradomain"])
    parser.add_argument("--input", type=str, default=None,
                        help="Input JSONL file (from prepare_*.py)")
    parser.add_argument("--sample_dir", type=str, default=None,
                        help="OminiRAG KG sample directory (alternative to --input)")
    parser.add_argument("--output", type=str, default="results/omini_eval.json",
                        help="Output JSON file path")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Number of samples (0 = all)")

    # Evaluation mode
    parser.add_argument("--eval_mode", type=str, default="generation",
                        choices=["generation", "pipeline"],
                        help="generation: use SelfRAGGeneration directly; "
                             "pipeline: build full LangGraph modular pipeline")

    # Model
    parser.add_argument("--model_name", type=str, default="selfrag/selfrag_llama2_7b")
    parser.add_argument("--download_dir", type=str,
                        default="/data1/ragworkspace/self-rag/model_cache")
    parser.add_argument("--dtype", type=str, default="half")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)

    # Self-RAG scoring
    parser.add_argument("--w_rel", type=float, default=1.0)
    parser.add_argument("--w_sup", type=float, default=1.0)
    parser.add_argument("--w_use", type=float, default=0.5)
    parser.add_argument("--use_seqscore", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--ndocs", type=int, default=10)

    # UltraDomain LLM judge
    parser.add_argument("--judge_model", type=str, default=None,
                        help="OpenAI model for UltraDomain LLM-as-judge "
                             "(e.g. gpt-4o-mini). Omit for F1-only evaluation.")

    # OminiRAG path (for importing benchmark adapters)
    parser.add_argument("--ominirag_path", type=str, default=None,
                        help="Path to OminiRAG root (auto-detected if on sys.path)")

    args = parser.parse_args()

    if not args.input and not args.sample_dir:
        parser.error("Must provide either --input (JSONL) or --sample_dir (KG sample)")

    # ---- Add OminiRAG to path if needed ----
    if args.ominirag_path:
        sys.path.insert(0, args.ominirag_path)

    # ---- Load Self-RAG model ----
    print(f"[INFO] Loading Self-RAG model: {args.model_name}")
    from vllm import LLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, cache_dir=args.download_dir,
    )
    model = LLM(
        model=args.model_name,
        download_dir=args.download_dir,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_logprobs=50000,
    )
    print("[INFO] Self-RAG model loaded")

    # ---- Build Self-RAG adapters ----
    from selfrag.constants import load_special_tokens
    from selfrag.adapters import SelfRAGGeneration, SelfRAGReranking

    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=True, use_utility=True,
    )

    generation = SelfRAGGeneration(
        model=model,
        rel_tokens=rel_tokens,
        grd_tokens=grd_tokens,
        ut_tokens=ut_tokens,
        w_rel=args.w_rel,
        w_sup=args.w_sup,
        w_use=args.w_use,
        use_seqscore=args.use_seqscore,
        max_new_tokens=args.max_new_tokens,
    )

    reranking = SelfRAGReranking(
        model=model,
        rel_tokens=rel_tokens,
        grd_tokens=grd_tokens,
        ut_tokens=ut_tokens,
        w_rel=args.w_rel,
        w_sup=args.w_sup,
        w_use=args.w_use,
        use_seqscore=args.use_seqscore,
        max_new_tokens=args.max_new_tokens,
    )

    pre_retrieval = PreRetrievedRetrieval()

    # ---- Load and convert data ----
    if args.dataset == "hotpotqa":
        if args.sample_dir:
            from benchmark.hotpotqa_adapter import (
                HotpotQABenchmarkAdapter,
                load_hotpotqa_sample,
            )
            data = load_hotpotqa_sample(args.sample_dir)
        else:
            from benchmark.hotpotqa_adapter import HotpotQABenchmarkAdapter
            raw_data = load_jsonl(args.input, args.num_samples)
            data = jsonl_to_omini_hotpotqa(raw_data)
        print(f"[INFO] Loaded {len(data)} HotPotQA examples")

    elif args.dataset == "ultradomain":
        if args.sample_dir:
            from benchmark.ultradomain_adapter import (
                UltraDomainBenchmarkAdapter,
                load_ultradomain_sample,
            )
            data = load_ultradomain_sample(args.sample_dir)
        else:
            from benchmark.ultradomain_adapter import UltraDomainBenchmarkAdapter
            raw_data = load_jsonl(args.input, args.num_samples)
            data = jsonl_to_omini_ultradomain(raw_data)
        print(f"[INFO] Loaded {len(data)} UltraDomain examples")

    # ---- Run evaluation ----
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.dataset == "hotpotqa":
        adapter = HotpotQABenchmarkAdapter()

        if args.eval_mode == "generation":
            print("[INFO] Running HotPotQA evaluation (generation mode)...")
            results = adapter.evaluate_generation(data, generation)
        else:
            print("[INFO] Building modular pipeline for HotPotQA (pipeline mode)...")
            from selfrag.modular_pipeline import build_selfrag_modular_graph
            graph = build_selfrag_modular_graph(
                retrieval=pre_retrieval,
                generation=generation,
                reranking=reranking,
            )
            results = adapter.evaluate_pipeline(data, graph)

        adapter.save_results(results, args.output)
        print(f"\n{'=' * 60}")
        print(f"HotPotQA Results (via OminiRAG adapter)")
        print(f"{'=' * 60}")
        print(f"  Exact Match:  {results.avg_em:.2f}%")
        print(f"  Token F1:     {results.avg_f1:.2f}%")
        print(f"  Num examples: {results.num_items}")
        print(f"  Saved to:     {args.output}")
        print(f"{'=' * 60}")

    elif args.dataset == "ultradomain":
        llm_judge = None
        if args.judge_model:
            print(f"[INFO] Creating LLM judge with {args.judge_model}")
            llm_judge = create_openai_judge(args.judge_model)

        adapter = UltraDomainBenchmarkAdapter(llm_complete=llm_judge)

        if args.eval_mode == "generation":
            print("[INFO] Running UltraDomain evaluation (generation mode)...")
            results = adapter.evaluate_generation(data, generation)
        else:
            print("[INFO] Building modular pipeline for UltraDomain (pipeline mode)...")
            from selfrag.modular_pipeline import build_selfrag_modular_graph
            graph = build_selfrag_modular_graph(
                retrieval=pre_retrieval,
                generation=generation,
                reranking=reranking,
            )
            results = adapter.evaluate_pipeline(data, graph)

        adapter.save_results(results, args.output)
        print(f"\n{'=' * 60}")
        print(f"UltraDomain Results (via OminiRAG adapter)")
        print(f"{'=' * 60}")
        print(f"  Token F1:             {results.avg_f1:.2f}%")
        print(f"  Comprehensiveness:    {results.avg_comprehensiveness:.2f}")
        print(f"  Diversity:            {results.avg_diversity:.2f}")
        print(f"  Empowerment:          {results.avg_empowerment:.2f}")
        print(f"  Avg Length (words):   {results.avg_length:.1f}")
        print(f"  Num examples:         {results.num_items}")
        print(f"  Saved to:             {args.output}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
