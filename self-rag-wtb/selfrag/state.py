from __future__ import annotations

from typing import Optional, TypedDict

from rag_contracts import GenerationResult, RetrievalResult


class SelfRAGModularState(TypedDict, total=False):
    # Input
    query: str

    # Stage outputs
    expanded_queries: list[str]
    retrieval_results: list[RetrievalResult]
    generation_result: GenerationResult

    # Error tracking
    error: Optional[str]
