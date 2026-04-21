"""WTB integration: register Self-RAG pipelines as WorkflowProject.

Usage:
    from wtb.sdk import WTBTestBench
    from wtb_integration import create_selfrag_query_project

    bench = WTBTestBench.create(mode="development")
    project = create_selfrag_query_project(llm_model, tokenizer, config)
    bench.register_project(project)
    result = bench.run(project.name, initial_state={"question": "Who wrote Hamlet?"})
"""
from functools import partial
from typing import Optional

from selfrag.config import SelfRAGConfig
from selfrag.constants import load_special_tokens
from selfrag.graph_query import build_query_graph
from selfrag.graph_query_longform import build_longform_query_graph
from selfrag.graph_index import build_index_graph
from selfrag.store import DocStore, VectorStore


def create_selfrag_query_project(
    llm_model,
    tokenizer,
    config: Optional[SelfRAGConfig] = None,
    doc_store=None,
    vector_store=None,
    retriever_tokenizer=None,
    retriever_model=None,
):
    """Create a WTB WorkflowProject for the Self-RAG short-form query pipeline.

    Args:
        llm_model: vLLM model instance (already loaded).
        tokenizer: HuggingFace tokenizer for loading special tokens.
        config: SelfRAGConfig (uses defaults if None).
        doc_store / vector_store: optional stores for real retrieval.
        retriever_tokenizer / retriever_model: optional Contriever for real retrieval.

    Returns:
        WorkflowProject ready for bench.register_project().
    """
    from wtb.sdk.workflow_project import WorkflowProject

    cfg = config or SelfRAGConfig()
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=True, use_utility=True,
    )

    def graph_factory():
        return build_query_graph(
            llm_model, ret_tokens, rel_tokens, grd_tokens, ut_tokens,
            cfg, doc_store, vector_store, retriever_tokenizer, retriever_model,
        )

    return WorkflowProject(
        name="selfrag_query",
        graph_factory=graph_factory,
    )


def create_selfrag_longform_project(
    llm_model,
    tokenizer,
    config: Optional[SelfRAGConfig] = None,
):
    """Create a WTB WorkflowProject for the Self-RAG long-form (beam search) pipeline.

    Args:
        llm_model: vLLM model instance (already loaded).
        tokenizer: HuggingFace tokenizer for loading special tokens.
        config: SelfRAGConfig (uses defaults if None).

    Returns:
        WorkflowProject ready for bench.register_project().
    """
    from wtb.sdk.workflow_project import WorkflowProject

    cfg = config or SelfRAGConfig()
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=True, use_utility=True,
    )

    def graph_factory():
        return build_longform_query_graph(
            llm_model, ret_tokens, rel_tokens, grd_tokens, ut_tokens, cfg,
        )

    return WorkflowProject(
        name="selfrag_longform",
        graph_factory=graph_factory,
    )


def create_selfrag_index_project(
    doc_store,
    vector_store,
    retriever_tokenizer,
    retriever_model,
):
    """Create a WTB WorkflowProject for the Self-RAG indexing pipeline.

    Args:
        doc_store: DocStore for passage persistence.
        vector_store: VectorStore for embedding persistence.
        retriever_tokenizer: Contriever tokenizer.
        retriever_model: Contriever model.

    Returns:
        WorkflowProject ready for bench.register_project().
    """
    from wtb.sdk.workflow_project import WorkflowProject

    def graph_factory():
        return build_index_graph(
            doc_store, vector_store, retriever_tokenizer, retriever_model,
        )

    return WorkflowProject(
        name="selfrag_index",
        graph_factory=graph_factory,
    )