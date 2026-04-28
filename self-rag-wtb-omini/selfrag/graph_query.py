"""Query pipeline: prompt -> decision -> (retrieve -> generate | no_retrieve) -> aggregate -> END

Full Self-RAG inference with pluggable retrieval.
"""
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from selfrag.state import QueryState
from selfrag.nodes.prompt_node import build_prompt_node
from selfrag.nodes.retrieval_node import build_retrieval_node
from selfrag.nodes.aggregate_node import build_aggregate_node


def _route_after_decision(state):
    """Conditional edge: choose retrieve or no-retrieve path."""
    return "retrieve" if state.get("do_retrieve") else "no_retrieval_generation"


def build_query_graph(
    llm_model,
    ret_tokens, rel_tokens, grd_tokens, ut_tokens,
    config,
    doc_store=None, vector_store=None,
    retriever_tokenizer=None, retriever_model=None,
    checkpointer=None,
):
    """Construct the query StateGraph.

    Args:
        llm_model: vLLM model for generation.
        ret_tokens / rel_tokens / grd_tokens / ut_tokens: reflection token maps.
        config: SelfRAGConfig instance.
        doc_store: DocStore (optional, for real retrieval).
        vector_store: VectorStore (optional, for real retrieval).
        retriever_tokenizer / retriever_model: Contriever (optional).
        checkpointer: Optional LangGraph checkpointer. If provided, compiles
            with it; otherwise returns uncompiled StateGraph (WTB-compatible).

    Returns:
        Uncompiled StateGraph (default) or compiled graph (if checkpointer given).
    """
    from selfrag.nodes.decision_node import build_decision_node
    from selfrag.nodes.generation_node import (
        build_evidence_generation_node,
        build_no_retrieval_generation_node,
    )

    g = StateGraph(QueryState)

    # ---- Nodes ----
    g.add_node("prepare_prompt", build_prompt_node())
    g.add_node("retrieval_decision", build_decision_node(llm_model, ret_tokens, config))
    g.add_node("retrieve", build_retrieval_node(
        config, doc_store, vector_store, retriever_tokenizer, retriever_model
    ))
    g.add_node("evidence_generation", build_evidence_generation_node(
        llm_model, rel_tokens, grd_tokens, ut_tokens, config
    ))
    g.add_node("no_retrieval_generation", build_no_retrieval_generation_node(llm_model, config))
    g.add_node("aggregate", build_aggregate_node())

    # ---- Edges ----
    g.add_edge(START, "prepare_prompt")
    g.add_edge("prepare_prompt", "retrieval_decision")
    g.add_conditional_edges(
        "retrieval_decision",
        _route_after_decision,
        {
            "retrieve": "retrieve",
            "no_retrieval_generation": "no_retrieval_generation",
        },
    )
    g.add_edge("retrieve", "evidence_generation")
    g.add_edge("evidence_generation", "aggregate")
    g.add_edge("no_retrieval_generation", "aggregate")
    g.add_edge("aggregate", END)

    if checkpointer:
        return g.compile(checkpointer=checkpointer)
    return g