"""Self-RAG LangGraph framework (WTB-compatible).

All graph builders return uncompiled StateGraph by default,
compatible with WTB's LangGraphStateAdapter.
Pass checkpointer=... to get a compiled graph for standalone use.

Graph builders (build_query_graph, build_longform_query_graph) require vllm
at call time. Import them explicitly when needed:
    from selfrag.graph_query import build_query_graph
"""
from selfrag.config import SelfRAGConfig
from selfrag.state import IndexState, QueryState, LongFormQueryState
from selfrag.store import DocStore, VectorStore