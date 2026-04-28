"""UltraDomain data preprocessing: chunk long contexts + Contriever retrieval.

UltraDomain (TommyChien/UltraDomain) contains book-level contexts (~80K tokens).
This script chunks each context into ~512-word windows with 50-word overlap,
encodes chunks with Contriever, and retrieves top-K passages per question.

Output format matches prepare_hotpotqa.py for unified downstream evaluation:
    {id, question, answer, answers, domain, evidences: [{title, text, score, rank}]}

Usage:
    python prepare_ultradomain.py \
        --data_dir /path/to/UltraDomain/jsonl_files \
        --output data/ultradomain_contriever.jsonl \
        --domains physics cs mathematics \
        --num_samples 200 \
        --batch_size 64 \
        --device cuda
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# =========================================================================
# Chunking
# =========================================================================

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[dict]:
    """Split text into word-level chunks with overlap.

    Args:
        text: The full document text.
        chunk_size: Number of words per chunk.
        overlap: Number of overlapping words between consecutive chunks.

    Returns:
        List of {"text": str, "chunk_idx": int} dicts.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        if not chunk_words:
            break
        chunks.append({
            "text": " ".join(chunk_words),
            "chunk_idx": len(chunks),
        })
        # Stop if we've consumed all words
        if i + chunk_size >= len(words):
            break

    return chunks


# =========================================================================
# Contriever encoding
# =========================================================================

def encode_texts(tokenizer, model, texts, batch_size=64, device="cuda"):
    """Encode texts with Contriever (mean-pool last hidden state)."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        embeddings = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        all_embeddings.append(embeddings.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)


def cosine_similarity(query_vec, doc_vecs):
    """Cosine similarity between a query vector and document vectors."""
    query_vec = query_vec.flatten()
    q_norm = np.linalg.norm(query_vec)
    d_norms = np.linalg.norm(doc_vecs, axis=1)
    if q_norm == 0:
        return np.zeros(len(doc_vecs))
    return (doc_vecs @ query_vec) / (d_norms * q_norm + 1e-10)


# =========================================================================
# Data loading (from local JSONL directory, matching OminiRAG UltraDomainAPI)
# =========================================================================

def load_ultradomain(data_dir: str, domains: list[str] | None = None) -> list[dict]:
    """Load UltraDomain JSONL files from a local directory.

    Each JSONL file is named <domain>.jsonl. If domains is None, load all.
    Standardizes the schema to match OminiRAG's _to_standard_item format.
    """
    root = Path(data_dir)
    all_jsonl = sorted(root.glob("*.jsonl"))
    if not all_jsonl:
        raise FileNotFoundError(f"No .jsonl files found in {data_dir}")

    available = [p.stem for p in all_jsonl]
    if domains:
        for d in domains:
            if d not in available:
                raise ValueError(f"Domain '{d}' not found. Available: {available}")
        selected = [p for p in all_jsonl if p.stem in domains]
    else:
        selected = all_jsonl

    items = []
    for path in selected:
        domain = path.stem
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                # Standardize fields
                query = (
                    raw.get("input")
                    or raw.get("question")
                    or raw.get("query")
                    or raw.get("prompt")
                    or raw.get("instruction")
                    or ""
                )
                answers = raw.get("answers")
                if isinstance(answers, list):
                    answer = answers[0] if answers else ""
                else:
                    answer = raw.get("answer") or raw.get("output") or ""
                    answers = [answer] if answer else []

                context = raw.get("context") or ""
                title = ""
                meta = raw.get("meta")
                if isinstance(meta, dict):
                    title = meta.get("title", "")

                items.append({
                    "id": f"{domain}::{idx}",
                    "domain": domain,
                    "question": query,
                    "answer": answer,
                    "answers": answers,
                    "context": context,
                    "title": title,
                    "length": raw.get("length", len(context)),
                })
    return items


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare UltraDomain with chunking + Contriever retrieval"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing UltraDomain .jsonl files")
    parser.add_argument("--output", type=str,
                        default="data/ultradomain_contriever.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--domains", nargs="*", default=None,
                        help="Domains to process (default: all)")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Max samples to process (0 = all)")
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="Words per chunk")
    parser.add_argument("--chunk_overlap", type=int, default=50,
                        help="Overlap words between chunks")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for Contriever encoding")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for Contriever model")
    parser.add_argument("--contriever_model", type=str,
                        default="facebook/contriever-msmarco",
                        help="Contriever model name or path")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of passages to retrieve per question")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    args = parser.parse_args()

    # ---- Load Contriever ----
    print(f"[INFO] Loading Contriever model: {args.contriever_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.contriever_model)
    model = AutoModel.from_pretrained(args.contriever_model).to(args.device)
    model.eval()
    print(f"[INFO] Contriever loaded on {args.device}")

    # ---- Load UltraDomain data ----
    print(f"[INFO] Loading UltraDomain from {args.data_dir}")
    if args.domains:
        print(f"[INFO] Target domains: {args.domains}")
    items = load_ultradomain(args.data_dir, args.domains)

    if args.num_samples > 0:
        import random
        rng = random.Random(args.seed)
        if args.num_samples < len(items):
            items = rng.sample(items, args.num_samples)
        print(f"[INFO] Sampled {len(items)} examples")
    else:
        print(f"[INFO] Processing all {len(items)} examples")

    # ---- Process each example ----
    os.makedirs(
        os.path.dirname(args.output) if os.path.dirname(args.output) else ".",
        exist_ok=True,
    )

    with open(args.output, "w", encoding="utf-8") as fout:
        for item in tqdm(items, desc="Processing"):
            context = item["context"]
            if not context.strip():
                continue

            # Chunk the long context
            chunks = chunk_text(
                context,
                chunk_size=args.chunk_size,
                overlap=args.chunk_overlap,
            )
            if not chunks:
                continue

            # Encode question and chunks with Contriever
            question = item["question"]
            chunk_texts = [c["text"] for c in chunks]

            query_emb = encode_texts(
                tokenizer, model, [question],
                batch_size=1, device=args.device,
            )
            doc_embs = encode_texts(
                tokenizer, model, chunk_texts,
                batch_size=args.batch_size, device=args.device,
            )

            # Rank by cosine similarity
            scores = cosine_similarity(query_emb[0], doc_embs)
            top_k = min(args.top_k, len(chunks))
            top_indices = np.argsort(-scores)[:top_k]

            evidences = []
            for rank, ci in enumerate(top_indices):
                evidences.append({
                    "title": item.get("title", ""),
                    "text": chunk_texts[ci],
                    "score": float(scores[ci]),
                    "rank": rank,
                    "chunk_idx": chunks[ci]["chunk_idx"],
                })

            record = {
                "id": item["id"],
                "question": question,
                "answer": item["answer"],
                "answers": item["answers"],
                "domain": item["domain"],
                "evidences": evidences,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[DONE] Saved to {args.output}")


if __name__ == "__main__":
    main()
