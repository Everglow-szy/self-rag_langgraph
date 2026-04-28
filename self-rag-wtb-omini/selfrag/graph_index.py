"""Indexing pipeline: chunk -> embedding -> END

Builds a VectorStore index from documents/passages.
"""
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from selfrag.state import IndexState
from selfrag.nodes.chunk_node import build_chunk_node
from selfrag.nodes.embedding_node import build_embedding_node


def build_index_graph(doc_store, vector_store, retriever_tokenizer, retriever_model,
                      checkpointer=None):
    """Construct the indexing StateGraph.

    Args:
        doc_store: DocStore instance for passage persistence.
        vector_store: VectorStore instance for embedding persistence.
        retriever_tokenizer: Contriever tokenizer.
        retriever_model: Contriever model.
        checkpointer: Optional LangGraph checkpointer. If provided, compiles
            with it; otherwise returns uncompiled StateGraph (WTB-compatible).

    Returns:
        Uncompiled StateGraph (default) or compiled graph (if checkpointer given).
    """
    g = StateGraph(IndexState)

    g.add_node("chunk", build_chunk_node(doc_store))
    g.add_node("embedding", build_embedding_node(vector_store, retriever_tokenizer, retriever_model))

    g.add_edge(START, "chunk")
    g.add_edge("chunk", "embedding")
    g.add_edge("embedding", END)

    if checkpointer:
        return g.compile(checkpointer=checkpointer)
    return g