"""Modular generation node: accepts a canonical ``rag_contracts.Generation`` via DI."""

from __future__ import annotations

from rag_contracts import Generation


def build_node(generation: Generation):
    """Build a generation node that calls the canonical Generation protocol."""

    async def node(state):
        query = state.get("query", state.get("question", ""))
        context = state.get("retrieval_results", [])
        result = generation.generate(query=query, context=context)
        return {
            "generation_result": result,
            "final_pred": result.output,
        }

    return node
