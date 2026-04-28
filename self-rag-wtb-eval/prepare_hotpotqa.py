"""HotPotQA data preprocessing: Contriever retrieval for each question.

Loads the HotPotQA distractor split, encodes context paragraphs with
Contriever, computes cosine similarity against the question, and saves
the top-10 passages per question as a JSONL file for downstream evaluation.

Usage:
    python prepare_hotpotqa.py \
        --split validation \
        --output data/hotpotqa_val_contriever.jsonl \
        --num_samples 500 \
        --batch_size 64 \
        --device cuda
"""

import argparse
import json
import os

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def encode_texts(tokenizer, model, texts, batch_size=64, device="cuda"):
    """Encode a list of texts with Contriever (mean-pool last hidden state)."""
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
    """Compute cosine similarity between a query vector and document vectors."""
    query_vec = query_vec.flatten()
    q_norm = np.linalg.norm(query_vec)
    d_norms = np.linalg.norm(doc_vecs, axis=1)
    if q_norm == 0:
        return np.zeros(len(doc_vecs))
    sims = (doc_vecs @ query_vec) / (d_norms * q_norm + 1e-10)
    return sims


def main():
    parser = argparse.ArgumentParser(description="Prepare HotPotQA with Contriever retrieval")
    parser.add_argument("--split", type=str, default="validation",
                        help="Dataset split: train / validation")
    parser.add_argument("--output", type=str, default="data/hotpotqa_val_contriever.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Number of samples to process (0 = all)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for Contriever encoding")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for Contriever model")
    parser.add_argument("--contriever_model", type=str,
                        default="facebook/contriever-msmarco",
                        help="Contriever model name or path")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of passages to retrieve per question")
    args = parser.parse_args()

    # ---- Load Contriever ----
    print(f"[INFO] Loading Contriever model: {args.contriever_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.contriever_model)
    model = AutoModel.from_pretrained(args.contriever_model).to(args.device)
    model.eval()
    print(f"[INFO] Contriever loaded on {args.device}")

    # ---- Load HotPotQA ----
    print(f"[INFO] Loading HotPotQA distractor split={args.split}")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=args.split)
    if args.num_samples > 0:
        ds = ds.select(range(min(args.num_samples, len(ds))))
    print(f"[INFO] Processing {len(ds)} examples")

    # ---- Process each example ----
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as fout:
        for idx in tqdm(range(len(ds)), desc="Processing"):
            example = ds[idx]
            question = example["question"]
            answer = example["answer"]
            qid = example["id"]

            # Extract context paragraphs: each is (title, list_of_sentences)
            titles = example["context"]["title"]
            sentences_list = example["context"]["sentences"]

            paragraphs = []
            for title, sents in zip(titles, sentences_list):
                text = " ".join(sents)
                paragraphs.append({"title": title, "text": text})

            if not paragraphs:
                continue

            # Encode question and paragraphs with Contriever
            doc_texts = [f"{p['title']}\n{p['text']}" for p in paragraphs]
            query_emb = encode_texts(tokenizer, model, [question],
                                     batch_size=1, device=args.device)
            doc_embs = encode_texts(tokenizer, model, doc_texts,
                                    batch_size=args.batch_size, device=args.device)

            # Compute cosine similarity and rank
            scores = cosine_similarity(query_emb[0], doc_embs)
            top_k = min(args.top_k, len(paragraphs))
            top_indices = np.argsort(-scores)[:top_k]

            evidences = []
            for rank, pi in enumerate(top_indices):
                evidences.append({
                    "title": paragraphs[pi]["title"],
                    "text": paragraphs[pi]["text"],
                    "score": float(scores[pi]),
                    "rank": rank,
                })

            record = {
                "id": qid,
                "question": question,
                "answer": answer,
                "type": example["type"],
                "level": example["level"],
                "supporting_facts": {
                    "title": example["supporting_facts"]["title"],
                    "sent_id": example["supporting_facts"]["sent_id"],
                },
                "evidences": evidences,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[DONE] Saved {len(ds)} examples to {args.output}")


if __name__ == "__main__":
    main()
