"""Modular Self-RAG pipeline using canonical rag_contracts components.

Graph topology:  query_processing -> retrieval -> reranking -> generation -> END

This mirrors the LongRAG modular graph and enables cross-project component
swaps.  Chunking and Embedding are offline/index-time and not part of this
query-time graph.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from rag_contracts import Generation, Query, Reranking, Retrieval

from selfrag.nodes.modular_generation_node import build_node as generation_node
from selfrag.nodes.modular_query_node import build_node as query_node
from selfrag.nodes.modular_reranking_node import build_node as reranking_node
from selfrag.nodes.modular_retrieval_node import build_node as retrieval_node
from selfrag.state import SelfRAGModularState


def build_selfrag_modular_graph(
    *,
    retrieval: Retrieval,
    generation: Generation,
    reranking: Reranking | None = None,
    query: Query | None = None,
):
    """Build the modular Self-RAG LangGraph pipeline.

    Four nodes:
      query_processing -> retrieval -> reranking -> generation -> END

    Always returns a compiled graph, matching LongRAG's ``build_graph()``
    behaviour.  For WTB integration (which needs an uncompiled
    ``StateGraph``), use :func:`build_selfrag_modular_state_graph` instead.

    Args:
        retrieval: Any ``rag_contracts.Retrieval`` implementation.
        generation: Any ``rag_contracts.Generation`` implementation.
        reranking: Optional ``rag_contracts.Reranking`` (defaults to IdentityReranking).
        query: Optional ``rag_contracts.Query`` (defaults to IdentityQuery).

    Returns:
        Compiled graph ready to ``.invoke()`` / ``.ainvoke()``.
    """
    return build_selfrag_modular_state_graph(
        retrieval=retrieval,
        generation=generation,
        reranking=reranking,
        query=query,
    ).compile()


def build_selfrag_modular_state_graph(
    *,
    retrieval: Retrieval,
    generation: Generation,
    reranking: Reranking | None = None,
    query: Query | None = None,
):
    """Build the modular Self-RAG pipeline as an uncompiled ``StateGraph``.

    Use this when you need the raw graph (e.g. for WTB's
    ``LangGraphStateAdapter``).  For direct execution prefer
    :func:`build_selfrag_modular_graph` which returns a compiled graph.
    """
    graph = StateGraph(SelfRAGModularState)

    graph.add_node("query_processing", query_node(query))
    graph.add_node("retrieval", retrieval_node(retrieval))
    graph.add_node("reranking", reranking_node(reranking))
    graph.add_node("generation", generation_node(generation))

    graph.set_entry_point("query_processing")
    graph.add_edge("query_processing", "retrieval")
    graph.add_edge("retrieval", "reranking")
    graph.add_edge("reranking", "generation")
    graph.add_edge("generation", END)

    return graph
