"""Modular reranking node: accepts a canonical ``rag_contracts.Reranking`` via DI."""

from __future__ import annotations

from rag_contracts import IdentityReranking, Reranking


def build_node(reranking: Reranking | None = None):
    """Build a reranking node that uses the canonical Reranking protocol."""
    _reranking = reranking or IdentityReranking()

    async def node(state):
        query = state.get("query", state.get("question", ""))
        results = state.get("retrieval_results", [])
        reranked = _reranking.rerank(query, results, top_k=10)
        return {"retrieval_results": reranked}

    return node
