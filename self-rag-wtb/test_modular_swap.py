"""Cross-project swap tests: verify that canonical components from LongRAG,
STORM, and Self-RAG can be freely swapped inside the modular Self-RAG pipeline.

Uses mock implementations to avoid heavy dependencies (vLLM, Contriever,
HuggingFace datasets, web APIs).
"""

import asyncio
import sys
import os
import types
from dataclasses import dataclass
from unittest.mock import MagicMock

# ── Inject fake vllm so selfrag.nodes can import SamplingParams ──────────────
fake_vllm = types.ModuleType("vllm")


@dataclass
class FakeSamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 100
    logprobs: int = 0


fake_vllm.SamplingParams = FakeSamplingParams
sys.modules["vllm"] = fake_vllm

if "torch" not in sys.modules:
    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = lambda: MagicMock(
        __enter__=lambda s: None, __exit__=lambda s, *a: None
    )
    sys.modules["torch"] = fake_torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from rag_contracts import (
    Chunking,
    Document,
    Embedding,
    Generation,
    GenerationResult,
    IdentityQuery,
    IdentityReranking,
    Query,
    QueryContext,
    Reranking,
    Retrieval,
    RetrievalResult,
)

from selfrag.adapters import (
    CanonicalToSelfRAGGeneration,
    CanonicalToSelfRAGRetrieval,
    SelfRAGChunking,
    SelfRAGEmbedding,
    SelfRAGGeneration,
    SelfRAGReranking,
    SelfRAGRetrieval,
    passage_to_retrieval_result,
    retrieval_result_to_passage,
)

# =============================================================================
# Mock components simulating LongRAG, STORM, and Self-RAG adapters
# =============================================================================


class MockLongRAGRetrieval:
    """Simulates ``HFDatasetRetrieval`` from LongRAG adapters."""

    def retrieve(self, queries: list[str], top_k: int = 10) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                source_id="longrag_doc_1",
                content="LongRAG retrieved passage about " + queries[0],
                score=0.95,
                title="LongRAG Doc",
                metadata={"source": "longrag"},
            )
        ][:top_k]


class MockLongRAGGeneration:
    """Simulates ``LongRAGGeneration`` from LongRAG adapters."""

    def generate(
        self, query: str, context: list[RetrievalResult], instruction: str = ""
    ) -> GenerationResult:
        ctx_text = "; ".join(r.content for r in context)
        return GenerationResult(
            output=f"LongRAG answer for '{query}' based on: {ctx_text}",
            citations=[r.source_id for r in context],
            metadata={"reader": "longrag"},
        )


class MockStormRetrieval:
    """Simulates ``StormRetrievalAdapter`` from STORM adapters."""

    def retrieve(self, queries: list[str], top_k: int = 10) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                source_id="https://storm-example.com/article",
                content="STORM web passage about " + queries[0],
                score=0.88,
                title="STORM Web Result",
                metadata={"source": "storm"},
            )
        ][:top_k]


class MockStormQuery:
    """Simulates ``StormQueryAdapter`` from STORM adapters."""

    def process(self, query: str, context: QueryContext) -> list[str]:
        return [query, f"What is the history of {query}?", f"Key facts about {query}"]


class MockStormReranking:
    """Simulates ``StormRerankingAdapter`` from STORM adapters."""

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int = 10
    ) -> list[RetrievalResult]:
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        return sorted_results[:top_k]


class MockStormGeneration:
    """Simulates ``StormGenerationAdapter`` from STORM adapters."""

    def generate(
        self, query: str, context: list[RetrievalResult], instruction: str = ""
    ) -> GenerationResult:
        ctx_text = "; ".join(r.content for r in context)
        return GenerationResult(
            output=f"## {query}\n\nSTORM article section based on: {ctx_text}",
            citations=[r.source_id for r in context],
            metadata={"writer": "storm"},
        )


class MockSelfRAGRetrieval:
    """Simulates ``SelfRAGRetrieval`` from selfrag/adapters.py."""

    def retrieve(self, queries: list[str], top_k: int = 10) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                source_id="selfrag_chunk_abc123",
                content="Self-RAG Contriever passage about " + queries[0],
                score=0.91,
                title="Self-RAG Doc",
                metadata={"source": "selfrag"},
            )
        ][:top_k]


class MockSelfRAGGeneration:
    """Simulates ``SelfRAGGeneration`` from selfrag/adapters.py."""

    def generate(
        self, query: str, context: list[RetrievalResult], instruction: str = ""
    ) -> GenerationResult:
        ctx_text = "; ".join(r.content for r in context)
        return GenerationResult(
            output=f"Self-RAG scored answer for '{query}' with context: {ctx_text}",
            citations=[r.source_id for r in context],
            metadata={"scorer": "selfrag", "selfrag_score": 1.85},
        )


class MockSelfRAGReranking:
    """Simulates ``SelfRAGReranking`` from selfrag/adapters.py."""

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int = 10
    ) -> list[RetrievalResult]:
        for r in results:
            r.metadata["selfrag_score"] = 1.5
        return results[:top_k]


# =============================================================================
# Helpers
# =============================================================================


def _run_modular_pipeline(
    retrieval: Retrieval,
    generation: Generation,
    reranking: Reranking | None = None,
    query: Query | None = None,
) -> dict:
    """Build and execute the Self-RAG modular pipeline, return final state."""
    from selfrag.modular_pipeline import build_selfrag_modular_graph

    compiled = build_selfrag_modular_graph(
        retrieval=retrieval,
        generation=generation,
        reranking=reranking,
        query=query,
    )
    result = asyncio.run(compiled.ainvoke({"query": "Who wrote Hamlet?"}))
    return result


def _run_longrag_pipeline(
    retrieval: Retrieval,
    generation: Generation,
    reranking: Reranking | None = None,
    query: Query | None = None,
) -> dict:
    """Build and execute the LongRAG pipeline, return final state."""
    from longRAG_example.longrag_langgraph.main_pipeline import build_graph

    compiled = build_graph(
        retrieval=retrieval,
        generation=generation,
        reranking=reranking,
        query=query,
    )
    result = asyncio.run(compiled.ainvoke({"query": "Who wrote Hamlet?"}))
    return result


# =============================================================================
# Tests: Protocol conformance of adapters
# =============================================================================


def test_selfrag_chunking_protocol():
    """SelfRAGChunking satisfies rag_contracts.Chunking."""
    chunker = SelfRAGChunking()
    assert isinstance(chunker, Chunking), "SelfRAGChunking must satisfy Chunking"
    docs = [
        Document(doc_id="d1", content="Hello world", metadata={"title": "Test"}),
    ]
    chunks = chunker.chunk(docs)
    assert len(chunks) == 1
    assert chunks[0].doc_id == "d1"
    assert chunks[0].content == "Hello world"
    print("[PASS] SelfRAGChunking satisfies Chunking protocol")


def test_selfrag_generation_protocol():
    """MockSelfRAGGeneration satisfies rag_contracts.Generation."""
    gen = MockSelfRAGGeneration()
    assert isinstance(gen, Generation), "Mock must satisfy Generation"
    print("[PASS] MockSelfRAGGeneration satisfies Generation protocol")


def test_selfrag_retrieval_protocol():
    """MockSelfRAGRetrieval satisfies rag_contracts.Retrieval."""
    ret = MockSelfRAGRetrieval()
    assert isinstance(ret, Retrieval), "Mock must satisfy Retrieval"
    print("[PASS] MockSelfRAGRetrieval satisfies Retrieval protocol")


def test_selfrag_reranking_protocol():
    """MockSelfRAGReranking satisfies rag_contracts.Reranking."""
    rr = MockSelfRAGReranking()
    assert isinstance(rr, Reranking), "Mock must satisfy Reranking"
    print("[PASS] MockSelfRAGReranking satisfies Reranking protocol")


# =============================================================================
# Tests: Protocol conformance of ACTUAL adapter classes (not mocks)
# =============================================================================


def test_actual_selfrag_chunking_protocol():
    """Actual SelfRAGChunking satisfies rag_contracts.Chunking."""
    assert isinstance(SelfRAGChunking(), Chunking)
    print("[PASS] SelfRAGChunking (actual) satisfies Chunking")


def test_actual_selfrag_embedding_protocol():
    """Actual SelfRAGEmbedding satisfies rag_contracts.Embedding."""
    adapter = SelfRAGEmbedding()
    assert isinstance(adapter, Embedding)
    vecs = adapter.embed(["hello"])
    assert vecs == [[]], "Should return empty vecs when no model is set"
    print("[PASS] SelfRAGEmbedding (actual) satisfies Embedding")


def test_actual_selfrag_retrieval_protocol():
    """Actual SelfRAGRetrieval satisfies rag_contracts.Retrieval."""
    adapter = SelfRAGRetrieval()
    assert isinstance(adapter, Retrieval)
    results = adapter.retrieve(["hello"])
    assert results == [], "Should return empty when no stores are set"
    print("[PASS] SelfRAGRetrieval (actual) satisfies Retrieval")


def test_actual_selfrag_reranking_protocol():
    """Actual SelfRAGReranking satisfies rag_contracts.Reranking."""
    adapter = SelfRAGReranking()
    assert isinstance(adapter, Reranking)
    dummy = [RetrievalResult(source_id="x", content="y", score=1.0)]
    reranked = adapter.rerank("q", dummy)
    assert len(reranked) == 1, "Should passthrough when no model is set"
    print("[PASS] SelfRAGReranking (actual) satisfies Reranking")


def test_actual_selfrag_generation_protocol():
    """Actual SelfRAGGeneration satisfies rag_contracts.Generation."""
    adapter = SelfRAGGeneration()
    assert isinstance(adapter, Generation)
    result = adapter.generate("q", [])
    assert result.output == "", "Should return empty when no model is set"
    print("[PASS] SelfRAGGeneration (actual) satisfies Generation")


# =============================================================================
# Tests: Conversion helpers
# =============================================================================


def test_passage_roundtrip():
    """passage_to_retrieval_result and retrieval_result_to_passage are inverses."""
    passage = {"title": "Shakespeare", "text": "He wrote Hamlet.", "doc_id": "d1"}
    rr = passage_to_retrieval_result(passage, idx=0)
    assert rr.title == "Shakespeare"
    assert rr.content == "He wrote Hamlet."
    assert rr.source_id == "d1"

    back = retrieval_result_to_passage(rr)
    assert back["title"] == passage["title"]
    assert back["text"] == passage["text"]
    assert back["doc_id"] == passage["doc_id"]
    print("[PASS] passage <-> RetrievalResult roundtrip")


# =============================================================================
# Tests: Modular pipeline (6 cross-project configurations)
# =============================================================================


def test_config1_selfrag_native():
    """Config 1: Self-RAG retrieval + Self-RAG generation (native modular)."""
    result = _run_modular_pipeline(
        retrieval=MockSelfRAGRetrieval(),
        generation=MockSelfRAGGeneration(),
    )
    assert "final_pred" in result
    assert "Self-RAG scored answer" in result["final_pred"]
    assert result.get("generation_result") is not None
    print(f"[PASS] Config 1 (SelfRAG native) -> '{result['final_pred'][:60]}...'")


def test_config2_longrag_retrieval_selfrag_gen():
    """Config 2: LongRAG retrieval + Self-RAG generation."""
    result = _run_modular_pipeline(
        retrieval=MockLongRAGRetrieval(),
        generation=MockSelfRAGGeneration(),
    )
    assert "final_pred" in result
    assert "Self-RAG scored answer" in result["final_pred"]
    assert "LongRAG retrieved" in result["final_pred"]
    print(f"[PASS] Config 2 (LongRAG ret + SelfRAG gen) -> '{result['final_pred'][:60]}...'")


def test_config3_selfrag_retrieval_longrag_gen():
    """Config 3: Self-RAG retrieval + LongRAG generation."""
    result = _run_modular_pipeline(
        retrieval=MockSelfRAGRetrieval(),
        generation=MockLongRAGGeneration(),
    )
    assert "final_pred" in result
    assert "LongRAG answer" in result["final_pred"]
    assert "Self-RAG Contriever" in result["final_pred"]
    print(f"[PASS] Config 3 (SelfRAG ret + LongRAG gen) -> '{result['final_pred'][:60]}...'")


def test_config4_storm_retrieval_selfrag_gen():
    """Config 4: STORM retrieval + Self-RAG generation."""
    result = _run_modular_pipeline(
        retrieval=MockStormRetrieval(),
        generation=MockSelfRAGGeneration(),
    )
    assert "final_pred" in result
    assert "Self-RAG scored answer" in result["final_pred"]
    assert "STORM web" in result["final_pred"]
    print(f"[PASS] Config 4 (STORM ret + SelfRAG gen) -> '{result['final_pred'][:60]}...'")


def test_config5_selfrag_gen_in_longrag_pipeline():
    """Config 5: Self-RAG generation as drop-in inside LongRAG pipeline."""
    result = _run_longrag_pipeline(
        retrieval=MockLongRAGRetrieval(),
        generation=MockSelfRAGGeneration(),
    )
    assert "generation_result" in result
    gen_result = result["generation_result"]
    assert "Self-RAG scored answer" in gen_result.output
    print(f"[PASS] Config 5 (SelfRAG gen in LongRAG pipe) -> '{gen_result.output[:60]}...'")


def test_config6_selfrag_reranking_in_longrag_pipeline():
    """Config 6: Self-RAG reranking + STORM generation inside LongRAG pipeline."""
    result = _run_longrag_pipeline(
        retrieval=MockLongRAGRetrieval(),
        generation=MockStormGeneration(),
        reranking=MockSelfRAGReranking(),
    )
    assert "generation_result" in result
    gen_result = result["generation_result"]
    assert "STORM article" in gen_result.output
    print(f"[PASS] Config 6 (SelfRAG rerank in LongRAG pipe) -> '{gen_result.output[:60]}...'")


def test_config7_storm_query_in_selfrag_modular():
    """Config 7: STORM query expansion + Self-RAG modular pipeline."""
    result = _run_modular_pipeline(
        retrieval=MockSelfRAGRetrieval(),
        generation=MockSelfRAGGeneration(),
        query=MockStormQuery(),
    )
    assert "final_pred" in result
    assert "Self-RAG scored answer" in result["final_pred"]
    expanded = result.get("expanded_queries", [])
    assert len(expanded) > 1, f"Expected query expansion, got {expanded}"
    print(f"[PASS] Config 7 (STORM query in SelfRAG modular) -> expanded to {len(expanded)} queries")


def test_config8_all_cross_project():
    """Config 8: STORM query + LongRAG retrieval + SelfRAG reranking + STORM generation."""
    result = _run_modular_pipeline(
        retrieval=MockLongRAGRetrieval(),
        generation=MockStormGeneration(),
        reranking=MockSelfRAGReranking(),
        query=MockStormQuery(),
    )
    assert "final_pred" in result
    assert "STORM article" in result["final_pred"]
    assert "LongRAG retrieved" in result["final_pred"]
    print(f"[PASS] Config 8 (all cross-project) -> '{result['final_pred'][:60]}...'")


# =============================================================================
# Tests: Reverse adapters
# =============================================================================


def test_reverse_retrieval_adapter():
    """CanonicalToSelfRAGRetrieval produces Self-RAG-compatible passages."""
    adapter = CanonicalToSelfRAGRetrieval(canonical_retrieval=MockLongRAGRetrieval())
    node_fn = adapter.as_retrieval_node()
    result = node_fn({"question": "Who wrote Hamlet?"})
    passages = result["retrieved_passages"]
    assert len(passages) > 0
    assert "title" in passages[0]
    assert "text" in passages[0]
    print(f"[PASS] CanonicalToSelfRAGRetrieval -> {len(passages)} passages")


def test_reverse_generation_adapter():
    """CanonicalToSelfRAGGeneration produces final_pred for Self-RAG state."""
    adapter = CanonicalToSelfRAGGeneration(canonical_generation=MockLongRAGGeneration())
    node_fn = adapter.as_aggregate_bypass_node()
    result = node_fn({
        "question": "Who wrote Hamlet?",
        "retrieved_passages": [
            {"title": "Shakespeare", "text": "He wrote Hamlet."},
        ],
    })
    assert "final_pred" in result
    assert "LongRAG answer" in result["final_pred"]
    print(f"[PASS] CanonicalToSelfRAGGeneration -> '{result['final_pred'][:60]}...'")


# =============================================================================
# Run all tests
# =============================================================================


if __name__ == "__main__":
    tests = [
        test_selfrag_chunking_protocol,
        test_selfrag_generation_protocol,
        test_selfrag_retrieval_protocol,
        test_selfrag_reranking_protocol,
        test_actual_selfrag_chunking_protocol,
        test_actual_selfrag_embedding_protocol,
        test_actual_selfrag_retrieval_protocol,
        test_actual_selfrag_reranking_protocol,
        test_actual_selfrag_generation_protocol,
        test_passage_roundtrip,
        test_config1_selfrag_native,
        test_config2_longrag_retrieval_selfrag_gen,
        test_config3_selfrag_retrieval_longrag_gen,
        test_config4_storm_retrieval_selfrag_gen,
        test_config5_selfrag_gen_in_longrag_pipeline,
        test_config6_selfrag_reranking_in_longrag_pipeline,
        test_config7_storm_query_in_selfrag_modular,
        test_config8_all_cross_project,
        test_reverse_retrieval_adapter,
        test_reverse_generation_adapter,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
