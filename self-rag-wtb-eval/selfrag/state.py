"""State schemas for Self-RAG LangGraph pipelines (WTB-compatible).

Defines state for all three pipelines:
  - IndexState:          chunking + embedding indexing
  - QueryState:          original Self-RAG vLLM inference (graph_query.py)
  - LongFormQueryState:  beam-search long-form generation
  - SelfRAGModularState: canonical rag_contracts modular pipeline
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from rag_contracts import GenerationResult, RetrievalResult


# ── Indexing pipeline ────────────────────────────────────────────────────

class IndexState(TypedDict, total=False):
    """State flowing through the indexing pipeline."""
    doc_id: str
    title: str
    text: str
    chunks: List[Dict[str, Any]]


# ── Original Self-RAG query pipeline (vLLM native) ──────────────────────

class QueryState(TypedDict, total=False):
    """State flowing through the query pipeline (graph_query.py)."""
    # ---- Inputs ----
    question: str
    gold_answers: List[str]
    evidences: List[Dict[str, Any]]   # pre-retrieved ctxs (optional)

    # ---- Config (per-query overrides) ----
    mode: str
    threshold: float
    w_rel: float
    w_sup: float
    w_use: float
    use_seqscore: bool
    max_new_tokens: int
    closed: bool
    ndocs: int

    # ---- Intermediate ----
    prompt: str
    do_retrieve: bool
    retrieval_decision_scores: Dict[str, float]
    retrieved_passages: List[Dict[str, Any]]
    no_retrieval_pred: str
    evidence_results: Dict[str, Dict[str, Any]]

    # ---- Output ----
    final_pred: str

    # ---- Multi-dataset support ----
    dataset: str                              # hotpotqa | ultradomain | popqa | ...
    answers: List[str]                        # multi-answer (UltraDomain)
    domain: str                               # UltraDomain domain label


# ── Long-form beam-search pipeline ──────────────────────────────────────

class LongFormQueryState(TypedDict, total=False):
    """State flowing through the long-form (beam search) query pipeline."""
    # ---- Inputs ----
    question: str
    task: str                                   # asqa | eli5 | factscore
    docs: List[Dict[str, Any]]
    gold_answers: List[str]

    # ---- Config ----
    mode: str
    threshold: float
    w_rel: float
    w_sup: float
    w_use: float
    use_seqscore: bool
    max_new_tokens: int
    ndocs: int
    beam_width: int
    max_depth: int
    ignore_cont: bool

    # ---- Intermediate ----
    prompt: str
    do_retrieve: bool
    prediction_tree: Dict[int, Dict[str, Any]]
    levels: Dict[int, List[int]]
    current_depth: int
    node_id_counter: int
    terminated: bool
    no_retrieval_pred: str

    # ---- Output ----
    final_pred: str
    final_output: str
    output_docs: List[Dict[str, Any]]
    intermediate: Dict[str, Any]


# ── Modular pipeline (rag_contracts canonical) ──────────────────────────

class SelfRAGModularState(TypedDict, total=False):
    """State for the canonical 4-node modular pipeline (modular_pipeline.py)."""
    # ---- Input ----
    query: str

    # ---- Stage outputs ----
    expanded_queries: list[str]
    retrieval_results: list[RetrievalResult]
    generation_result: GenerationResult

    # ---- Error tracking ----
    error: Optional[str]

    # ---- Benchmark adapter support (OminiRAG evaluate_pipeline) ----
    query_id: str
    answers: list[str]
    test_data_name: str
    domain: str
