"""Indexing pipeline: chunk -> embedding -> END

Builds a VectorStore index from documents/passages.
"""
from langgraph.graph import END, START, StateGraph

from langgraph_nodes.state import IndexState
from langgraph_nodes.nodes.chunk_node import build_chunk_node
from langgraph_nodes.nodes.embedding_node import build_embedding_node


def build_index_graph(doc_store, vector_store, retriever_tokenizer, retriever_model):
    """Construct the indexing StateGraph.

    Args:
        doc_store: DocStore instance for passage persistence.
        vector_store: VectorStore instance for embedding persistence.
        retriever_tokenizer: Contriever tokenizer.
        retriever_model: Contriever model.

    Returns:
        Compiled LangGraph graph.
    """
    g = StateGraph(IndexState)

    g.add_node("chunk", build_chunk_node(doc_store))
    g.add_node("embedding", build_embedding_node(vector_store, retriever_tokenizer, retriever_model))

    g.add_edge(START, "chunk")
    g.add_edge("chunk", "embedding")
    g.add_edge("embedding", END)

    return g.compile()