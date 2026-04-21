"""State schemas for Self-RAG LangGraph pipelines (WTB-compatible)."""
from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class IndexState(TypedDict, total=False):
    """State flowing through the indexing pipeline."""
    doc_id: str
    title: str
    text: str
    chunks: List[Dict[str, Any]]   # [{chunk_id, title, text, doc_id}]


class QueryState(TypedDict, total=False):
    """State flowing through the query pipeline."""
    # ---- Inputs ----
    question: str
    gold_answers: List[str]
    evidences: List[Dict[str, Any]]   # pre-retrieved ctxs (optional, for compat)

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
    retrieved_passages: List[Dict[str, Any]]   # from retrieval node
    no_retrieval_pred: str
    evidence_results: Dict[str, Dict[str, Any]]

    # ---- Output ----
    final_pred: str


class LongFormQueryState(TypedDict, total=False):
    """State flowing through the long-form (beam search) query pipeline."""
    # ---- Inputs ----
    question: str
    task: str                                   # "asqa" | "eli5" | "factscore"
    docs: List[Dict[str, Any]]                  # pre-retrieved ctxs from ALCE data
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
    final_output: str          # with citations
    output_docs: List[Dict[str, Any]]
    intermediate: Dict[str, Any]