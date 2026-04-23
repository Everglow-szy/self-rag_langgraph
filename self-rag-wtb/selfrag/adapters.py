"""Adapters between Self-RAG internals and canonical rag_contracts.

Forward direction: wrap Self-RAG components to satisfy canonical protocols.
Reverse direction: wrap canonical protocols for use inside Self-RAG's graph.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from rag_contracts import (
    Chunk,
    Document,
    GenerationResult,
    QueryContext,
    RetrievalResult,
)

from selfrag.constants import control_tokens


# =============================================================================
# Conversion helpers
# =============================================================================


def passage_to_retrieval_result(p: dict, idx: int = 0) -> RetrievalResult:
    """Convert a Self-RAG passage dict ``{title, text, ...}`` to canonical type."""
    return RetrievalResult(
        source_id=p.get("doc_id", f"selfrag_passage_{idx}"),
        content=p.get("text", ""),
        score=p.get("retrieval_score", 0.0),
        title=p.get("title", ""),
        metadata={k: v for k, v in p.items() if k not in ("title", "text", "doc_id")},
    )


def retrieval_result_to_passage(r: RetrievalResult) -> dict:
    """Convert a canonical ``RetrievalResult`` back to Self-RAG passage dict."""
    return {
        "title": r.title,
        "text": r.content,
        "doc_id": r.source_id,
        "retrieval_score": r.score,
    }


def _postprocess(answer: str) -> str:
    """Remove Self-RAG special tokens from generated text."""
    for t in control_tokens:
        answer = answer.replace(t, "")
    return (
        answer.replace("</s>", "")
        .replace("\n", "")
        .replace("<|endoftext|>", "")
    )


def compute_selfrag_score(
    pred: Any,
    rel_tokens: Dict[str, int],
    grd_tokens: Optional[Dict[str, int]] = None,
    ut_tokens: Optional[Dict[str, int]] = None,
    w_rel: float = 1.0,
    w_sup: float = 1.0,
    w_use: float = 0.5,
    use_seqscore: bool = False,
) -> float:
    """Compute the Self-RAG composite score for a single prediction.

    Combines relevance (ISREL), grounding (ISSUP), and utility (ISUSE)
    signals from the model's logprobs.  Used by both ``SelfRAGReranking``
    and ``SelfRAGGeneration`` to avoid duplicated scoring logic.
    """
    token_ids = pred.outputs[0].token_ids
    logprobs = pred.outputs[0].logprobs
    seq_score = pred.outputs[0].cumulative_logprob / max(len(token_ids), 1)

    # ISREL (relevance) -- first token position
    rel_dict: Dict[str, float] = {}
    for tok, tid in rel_tokens.items():
        prob = logprobs[0].get(tid, -100) if logprobs else -100
        rel_dict[tok] = float(np.exp(float(prob)))
    rel_sum = float(np.sum(list(rel_dict.values()))) or 1.0
    relevance_score = rel_dict.get("[Relevant]", 0.0) / rel_sum

    # ISSUP (grounding) -- position of first grounding token
    ground_score = 0.0
    if grd_tokens is not None:
        grd_vals = list(grd_tokens.values())
        grd_idx = None
        for i, tid in enumerate(token_ids):
            if tid in grd_vals:
                grd_idx = i
                break
        if grd_idx is not None and grd_idx < len(logprobs):
            grd_dict: Dict[str, float] = {}
            for tok, tid in grd_tokens.items():
                prob = logprobs[grd_idx].get(tid, -100)
                grd_dict[tok] = float(np.exp(float(prob)))
            if len(grd_dict) == 3:
                gt_sum = float(np.sum(list(grd_dict.values()))) or 1.0
                ground_score = (
                    grd_dict["[Fully supported]"] / gt_sum
                    + 0.5 * grd_dict["[Partially supported]"] / gt_sum
                )

    # ISUSE (utility) -- position of first utility token
    utility_score = 0.0
    if ut_tokens is not None:
        ut_vals = list(ut_tokens.values())
        ut_indices = [i for i, tid in enumerate(token_ids) if tid in ut_vals]
        if ut_indices:
            idx = ut_indices[0]
            if idx < len(logprobs):
                ut_dict: Dict[str, float] = {}
                for tok, tid in ut_tokens.items():
                    prob = logprobs[idx].get(tid, -100)
                    ut_dict[tok] = float(np.exp(float(prob)))
                if len(ut_dict) == 5:
                    ut_sum = float(np.sum(list(ut_dict.values()))) or 1.0
                    ut_scale = [-1, -0.5, 0, 0.5, 1]
                    utility_score = float(np.sum([
                        ut_scale[i] * (ut_dict["[Utility:{}]".format(i + 1)] / ut_sum)
                        for i in range(5)
                    ]))

    if use_seqscore:
        return (
            float(np.exp(seq_score))
            + w_rel * relevance_score
            + w_sup * ground_score
            + w_use * utility_score
        )
    return (
        w_rel * relevance_score
        + w_sup * ground_score
        + w_use * utility_score
    )


# =============================================================================
# Forward adapters: Self-RAG implementation -> canonical protocol
# =============================================================================


@dataclass
class SelfRAGChunking:
    """``rag_contracts.Chunking`` wrapping Self-RAG's MD5-based single-passage chunking.

    Optionally persists to a ``DocStore`` for later retrieval.
    """

    doc_store: Any = None

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            title = doc.metadata.get("title", "")
            text = doc.content
            chunk_id = hashlib.md5(f"{title}||{text}".encode()).hexdigest()

            if self.doc_store is not None:
                self.doc_store.upsert(chunk_id, title=title, text=text, doc_id=doc.doc_id)

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    content=text,
                    metadata={"title": title},
                )
            )
        return chunks


@dataclass
class SelfRAGEmbedding:
    """``rag_contracts.Embedding`` wrapping Contriever mean-pool encoding.

    Optionally persists to a ``VectorStore``.
    """

    retriever_tokenizer: Any = None
    retriever_model: Any = None
    vector_store: Any = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.retriever_model is None or self.retriever_tokenizer is None:
            return [[] for _ in texts]

        import torch

        inputs = self.retriever_tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        device = next(self.retriever_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.retriever_model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        embeddings = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        vecs = embeddings.cpu().numpy()

        if self.vector_store is not None:
            ids = [hashlib.md5(t.encode()).hexdigest() for t in texts]
            self.vector_store.upsert(ids, vecs)

        return vecs.tolist()


@dataclass
class SelfRAGRetrieval:
    """``rag_contracts.Retrieval`` wrapping Contriever encode + VectorStore cosine search."""

    doc_store: Any = None
    vector_store: Any = None
    retriever_tokenizer: Any = None
    retriever_model: Any = None

    def _encode_query(self, text: str) -> np.ndarray:
        import torch

        inputs = self.retriever_tokenizer(
            [text], padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        device = next(self.retriever_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.retriever_model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        emb = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        return emb.cpu().numpy()[0]

    def retrieve(
        self, queries: list[str], top_k: int = 10
    ) -> list[RetrievalResult]:
        if (
            self.retriever_model is None
            or self.vector_store is None
            or self.doc_store is None
        ):
            return []

        results: list[RetrievalResult] = []
        for q in queries:
            query_vec = self._encode_query(q)
            hits = self.vector_store.search(query_vec, top_k=top_k)
            for chunk_id, score in hits:
                doc = self.doc_store.get(chunk_id)
                if doc:
                    results.append(
                        RetrievalResult(
                            source_id=chunk_id,
                            content=doc["text"],
                            score=score,
                            title=doc.get("title", ""),
                            metadata={"doc_id": doc.get("doc_id", chunk_id)},
                        )
                    )
        return results[:top_k]


@dataclass
class SelfRAGReranking:
    """``rag_contracts.Reranking`` using Self-RAG's evidence-generation scoring.

    Generates per-passage answers internally to compute relevance, grounding,
    and utility scores, then returns passages reordered by those scores.
    The generated text is discarded; only the ranking is used.
    """

    model: Any = None
    rel_tokens: Dict[str, int] = field(default_factory=dict)
    grd_tokens: Optional[Dict[str, int]] = None
    ut_tokens: Optional[Dict[str, int]] = None
    w_rel: float = 1.0
    w_sup: float = 1.0
    w_use: float = 0.5
    use_seqscore: bool = False
    max_new_tokens: int = 100

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int = 10
    ) -> list[RetrievalResult]:
        if self.model is None or not results or not self.rel_tokens:
            return results[:top_k]

        from selfrag.constants import PROMPT_DICT
        from vllm import SamplingParams

        prompt = PROMPT_DICT["prompt_no_input"].format_map({"instruction": query})
        passages = [retrieval_result_to_passage(r) for r in results]

        augmented = [
            prompt + "[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(
                p.get("title", ""), p.get("text", "")
            )
            for p in passages
        ]
        sp = SamplingParams(
            temperature=0.0, top_p=1.0,
            max_tokens=self.max_new_tokens,
            logprobs=5000,
        )
        preds = self.model.generate(augmented, sp)

        scored: list[tuple[int, float]] = []
        for p_idx, pred in enumerate(preds):
            score = compute_selfrag_score(
                pred, self.rel_tokens, self.grd_tokens, self.ut_tokens,
                self.w_rel, self.w_sup, self.w_use, self.use_seqscore,
            )
            scored.append((p_idx, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        reranked = []
        for p_idx, score in scored[:top_k]:
            r = results[p_idx]
            reranked.append(
                RetrievalResult(
                    source_id=r.source_id,
                    content=r.content,
                    score=score,
                    title=r.title,
                    metadata={**r.metadata, "selfrag_score": score},
                )
            )
        return reranked


@dataclass
class SelfRAGGeneration:
    """``rag_contracts.Generation`` wrapping evidence_generation + aggregate.

    For each context passage the model generates a candidate answer and scores
    it using Self-RAG's logprob-based relevance/grounding/utility signals.
    The highest-scoring answer is returned as the canonical ``GenerationResult``.
    """

    model: Any = None
    rel_tokens: Dict[str, int] = field(default_factory=dict)
    grd_tokens: Optional[Dict[str, int]] = None
    ut_tokens: Optional[Dict[str, int]] = None
    w_rel: float = 1.0
    w_sup: float = 1.0
    w_use: float = 0.5
    use_seqscore: bool = False
    max_new_tokens: int = 100

    def generate(
        self,
        query: str,
        context: list[RetrievalResult],
        instruction: str = "",
    ) -> GenerationResult:
        if self.model is None:
            return GenerationResult(output="", citations=[])

        from selfrag.constants import PROMPT_DICT
        from vllm import SamplingParams

        prompt = PROMPT_DICT["prompt_no_input"].format_map({"instruction": query})
        passages = [retrieval_result_to_passage(r) for r in context]

        if not passages:
            sp = SamplingParams(
                temperature=0.0, top_p=1.0,
                max_tokens=self.max_new_tokens,
            )
            preds = self.model.generate([prompt + "[No Retrieval]"], sp)
            text = _postprocess(preds[0].outputs[0].text)
            return GenerationResult(output=text, citations=[])

        augmented = [
            prompt + "[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(
                p.get("title", ""), p.get("text", "")
            )
            for p in passages
        ]
        sp = SamplingParams(
            temperature=0.0, top_p=1.0,
            max_tokens=self.max_new_tokens,
            logprobs=5000,
        )
        preds = self.model.generate(augmented, sp)

        best_idx = 0
        best_score = float("-inf")
        for p_idx, pred in enumerate(preds):
            score = compute_selfrag_score(
                pred, self.rel_tokens, self.grd_tokens, self.ut_tokens,
                self.w_rel, self.w_sup, self.w_use, self.use_seqscore,
            )
            if score > best_score:
                best_score = score
                best_idx = p_idx

        best_text = _postprocess(preds[best_idx].outputs[0].text)
        if best_text and best_text[0] in ("#", ":"):
            best_text = best_text[1:]

        citations = [context[best_idx].source_id] if context else []
        return GenerationResult(
            output=best_text,
            citations=citations,
            metadata={
                "selfrag_score": best_score,
                "passage_index": best_idx,
                "num_passages_scored": len(preds),
            },
        )


# =============================================================================
# Reverse adapters: canonical protocol -> Self-RAG internal
# =============================================================================


@dataclass
class CanonicalToSelfRAGRetrieval:
    """Wraps a ``rag_contracts.Retrieval`` to produce Self-RAG's
    ``retrieved_passages`` format (``list[dict]`` with *title* and *text* keys).

    Drop this into ``build_retrieval_node`` as a replacement for VectorStore
    retrieval, or use directly in the modular pipeline.
    """

    canonical_retrieval: Any
    default_top_k: int = 5

    def as_retrieval_node(self):
        """Return a Self-RAG-compatible node function for ``graph_query.py``."""
        canonical = self.canonical_retrieval
        top_k = self.default_top_k

        def retrieval_node(state: dict) -> dict:
            question = state.get("question", state.get("query", ""))
            ndocs = state.get("ndocs", top_k)
            results: list[RetrievalResult] = canonical.retrieve(
                [question], top_k=ndocs
            )
            passages = [retrieval_result_to_passage(r) for r in results]
            return {"retrieved_passages": passages}

        return retrieval_node


@dataclass
class CanonicalToSelfRAGGeneration:
    """Wraps a ``rag_contracts.Generation`` so it can be used inside the
    original Self-RAG graph in place of evidence_generation + aggregate.

    The node takes ``retrieved_passages`` from state, converts them to
    canonical types, calls the canonical generator, and writes ``final_pred``.
    """

    canonical_generation: Any

    def as_aggregate_bypass_node(self):
        """Return a node that replaces both evidence_generation and aggregate.

        Wire this as a single node that goes straight to END.
        """
        gen = self.canonical_generation

        def generation_node(state: dict) -> dict:
            question = state.get("question", state.get("query", ""))
            passages = state.get("retrieved_passages") or state.get("evidences") or []
            context = [
                passage_to_retrieval_result(p, idx=i) for i, p in enumerate(passages)
            ]
            result: GenerationResult = gen.generate(query=question, context=context)
            return {
                "final_pred": result.output,
                "evidence_results": {},
            }

        return generation_node
