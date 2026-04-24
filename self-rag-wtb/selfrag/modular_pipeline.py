"""Self-RAG modular pipeline rebuilt with LangGraph and canonical rag_contracts.

Provides the same 4-node topology as LongRAG and LightRAG pipelines:
  query_processing -> retrieval -> reranking -> generation -> END

All components are injected via dependency injection and must satisfy the
canonical rag_contracts protocols.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from rag_contracts import Generation, Query, Reranking, Retrieval

from .nodes.generation_node import build_node as generation_node
from .nodes.query_node import build_node as query_node
from .nodes.reranking_node import build_node as reranking_node
from .nodes.retrieval_node import build_node as retrieval_node
from .state import SelfRAGModularState


def build_selfrag_modular_graph(
    *,
    retrieval: Retrieval,
    generation: Generation,
    reranking: Reranking | None = None,
    query: Query | None = None,
):
    """Build the Self-RAG modular LangGraph pipeline with DI.

    Four nodes:
      query_processing -> retrieval -> reranking -> generation -> END

    Identical topology to ``build_graph()`` (LongRAG) and
    ``build_query_graph()`` (LightRAG), enabling cross-framework
    component swapping.
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
    return graph.compile()
