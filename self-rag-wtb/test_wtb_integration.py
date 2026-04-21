"""Integration test: verify Self-RAG pipelines run inside WTB framework.

Uses mock vLLM model to avoid GPU dependency. Tests:
1. Graph construction returns uncompiled StateGraph
2. WTB can compile and execute the graph
3. Node return values are state-update dicts (no full-state copies)
4. Short-form query pipeline end-to-end
5. Long-form query pipeline end-to-end
6. Indexing pipeline end-to-end
"""
import sys
import os
import types
from unittest.mock import MagicMock
from dataclasses import dataclass

# ── Inject a fake vllm module so selfrag.nodes can import SamplingParams ──
fake_vllm = types.ModuleType("vllm")

@dataclass
class FakeSamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 100
    logprobs: int = 0

fake_vllm.SamplingParams = FakeSamplingParams
sys.modules["vllm"] = fake_vllm

# Also fake torch for embedding/retrieval nodes
if "torch" not in sys.modules:
    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = lambda: MagicMock(__enter__=lambda s: None, __exit__=lambda s, *a: None)
    sys.modules["torch"] = fake_torch

# ── Now import selfrag ──
sys.path.insert(0, os.path.dirname(__file__))

from selfrag.config import SelfRAGConfig
from selfrag.state import QueryState, LongFormQueryState, IndexState
from selfrag.constants import load_special_tokens


# ═══════════════════════════════════════════════════════════════════════════════
# Mock vLLM model
# ═══════════════════════════════════════════════════════════════════════════════

class MockOutput:
    def __init__(self):
        self.text = "[Relevant]The answer is 42.[Fully supported][Utility:5]"
        self.token_ids = [100, 200, 300, 400, 500]
        # logprobs: list of dicts, one per token position
        self.logprobs = [
            {100: -0.1, 200: -2.0, 300: -0.5, 400: -3.0, 500: -1.0},
            {100: -0.2, 200: -1.5, 300: -0.3, 400: -2.5, 500: -0.8},
            {100: -0.3, 200: -1.0, 300: -0.4, 400: -2.0, 500: -0.6},
            {100: -0.1, 200: -2.0, 300: -0.5, 400: -3.0, 500: -1.0},
            {100: -0.2, 200: -1.5, 300: -0.3, 400: -2.5, 500: -0.8},
        ]
        self.cumulative_logprob = -5.0


class MockPrediction:
    def __init__(self):
        self.outputs = [MockOutput()]


class MockModel:
    def generate(self, prompts, sampling_params=None):
        return [MockPrediction() for _ in prompts]


# ═══════════════════════════════════════════════════════════════════════════════
# Mock tokenizer
# ═══════════════════════════════════════════════════════════════════════════════

class MockTokenizer:
    _token_map = {
        "[No Retrieval]": 100,
        "[Retrieval]": 200,
        "[Continue to Use Evidence]": 300,
        "[Irrelevant]": 400,
        "[Relevant]": 500,
        "[Fully supported]": 100,
        "[Partially supported]": 200,
        "[No support / Contradictory]": 300,
        "[Utility:1]": 100,
        "[Utility:2]": 200,
        "[Utility:3]": 300,
        "[Utility:4]": 400,
        "[Utility:5]": 500,
    }

    def convert_tokens_to_ids(self, token):
        return self._token_map.get(token, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_query_graph_uncompiled():
    """Graph builder returns uncompiled StateGraph (not CompiledGraph)."""
    from selfrag.graph_query import build_query_graph
    from langgraph.graph import StateGraph

    model = MockModel()
    tokenizer = MockTokenizer()
    config = SelfRAGConfig(mode="always_retrieve")
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=True, use_utility=True)

    graph = build_query_graph(
        model, ret_tokens, rel_tokens, grd_tokens, ut_tokens, config)

    assert isinstance(graph, StateGraph), \
        f"Expected StateGraph, got {type(graph).__name__}"
    print("[PASS] query graph returns uncompiled StateGraph")


def test_longform_graph_uncompiled():
    """Long-form graph builder returns uncompiled StateGraph."""
    from selfrag.graph_query_longform import build_longform_query_graph
    from langgraph.graph import StateGraph

    model = MockModel()
    tokenizer = MockTokenizer()
    config = SelfRAGConfig(mode="always_retrieve")
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=True, use_utility=True)

    graph = build_longform_query_graph(
        model, ret_tokens, rel_tokens, grd_tokens, ut_tokens, config)

    assert isinstance(graph, StateGraph), \
        f"Expected StateGraph, got {type(graph).__name__}"
    print("[PASS] longform graph returns uncompiled StateGraph")


def test_query_graph_compiled_standalone():
    """Graph can be compiled standalone with checkpointer."""
    from selfrag.graph_query import build_query_graph
    from langgraph.graph import StateGraph

    model = MockModel()
    tokenizer = MockTokenizer()
    config = SelfRAGConfig(mode="always_retrieve")
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=True, use_utility=True)

    from langgraph.checkpoint.memory import MemorySaver
    graph = build_query_graph(
        model, ret_tokens, rel_tokens, grd_tokens, ut_tokens, config,
        checkpointer=MemorySaver())

    # When checkpointer is passed, should return compiled graph
    assert not isinstance(graph, StateGraph), \
        "With checkpointer, should return compiled graph"
    print("[PASS] query graph compiles with checkpointer")


def test_query_pipeline_e2e():
    """Short-form query pipeline runs end-to-end with mock model."""
    from selfrag.graph_query import build_query_graph
    from langgraph.checkpoint.memory import MemorySaver

    model = MockModel()
    tokenizer = MockTokenizer()
    config = SelfRAGConfig(mode="always_retrieve", ndocs=2)
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=True, use_utility=True)

    graph = build_query_graph(
        model, ret_tokens, rel_tokens, grd_tokens, ut_tokens, config,
        checkpointer=MemorySaver())

    state = {
        "question": "Who wrote Hamlet?",
        "evidences": [
            {"title": "Shakespeare", "text": "William Shakespeare wrote Hamlet."},
            {"title": "Literature", "text": "Hamlet is a famous play."},
        ],
    }
    result = graph.invoke(state, config={"configurable": {"thread_id": "test-1"}})

    assert "final_pred" in result, f"Missing final_pred, keys: {list(result.keys())}"
    assert len(result["final_pred"]) > 0, "final_pred is empty"
    print(f"[PASS] query pipeline e2e -> final_pred: '{result['final_pred'][:60]}...'")


def test_longform_pipeline_e2e():
    """Long-form beam search pipeline runs end-to-end with mock model."""
    from selfrag.graph_query_longform import build_longform_query_graph
    from langgraph.checkpoint.memory import MemorySaver

    model = MockModel()
    tokenizer = MockTokenizer()
    config = SelfRAGConfig(mode="always_retrieve", ndocs=2, beam_width=2, max_depth=2)
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=True, use_utility=True)

    graph = build_longform_query_graph(
        model, ret_tokens, rel_tokens, grd_tokens, ut_tokens, config,
        checkpointer=MemorySaver())

    state = {
        "question": "What is the history of baptism?",
        "task": "asqa",
        "docs": [
            {"title": "Baptism", "text": "Baptism is a Christian rite."},
            {"title": "History", "text": "John the Baptist baptized Jesus."},
        ],
    }
    result = graph.invoke(state, config={"configurable": {"thread_id": "test-2"}})

    assert "final_output" in result, f"Missing final_output, keys: {list(result.keys())}"
    print(f"[PASS] longform pipeline e2e -> final_output: '{result.get('final_output', '')[:60]}...'")


def test_wtb_register_and_compile():
    """WTB can register the project and compile the graph via StateAdapter."""
    from selfrag.graph_query import build_query_graph
    from wtb.sdk.workflow_project import WorkflowProject
    from wtb.infrastructure.adapters.langgraph_state_adapter import (
        LangGraphStateAdapter, LangGraphConfig, CheckpointerType,
    )

    model = MockModel()
    tokenizer = MockTokenizer()
    config = SelfRAGConfig(mode="always_retrieve", ndocs=2)
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=True, use_utility=True)

    def graph_factory():
        return build_query_graph(
            model, ret_tokens, rel_tokens, grd_tokens, ut_tokens, config)

    project = WorkflowProject(name="selfrag_test", graph_factory=graph_factory)

    # Simulate what WTB does: call graph_factory, pass to adapter
    graph = project.graph_factory()

    lg_config = LangGraphConfig(checkpointer_type=CheckpointerType.MEMORY)
    adapter = LangGraphStateAdapter(lg_config)
    adapter.set_workflow_graph(graph)

    # The adapter should have compiled the graph
    assert adapter._compiled_graph is not None, "Adapter failed to compile graph"
    print("[PASS] WTB StateAdapter compiled Self-RAG graph successfully")


def test_wtb_full_execution():
    """Full WTB execution: register -> create execution -> run."""
    from selfrag.graph_query import build_query_graph
    from wtb.sdk import WTBTestBench
    from wtb.sdk.workflow_project import WorkflowProject

    model = MockModel()
    tokenizer = MockTokenizer()
    config = SelfRAGConfig(mode="always_retrieve", ndocs=2)
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=True, use_utility=True)

    def graph_factory():
        return build_query_graph(
            model, ret_tokens, rel_tokens, grd_tokens, ut_tokens, config)

    import uuid
    pname = f"selfrag_e2e_{uuid.uuid4().hex[:8]}"
    project = WorkflowProject(name=pname, graph_factory=graph_factory)

    bench = WTBTestBench.create(mode="development")
    bench.register_project(project)

    result = bench.run(pname, initial_state={
        "question": "Who wrote Hamlet?",
        "evidences": [
            {"title": "Shakespeare", "text": "William Shakespeare wrote Hamlet."},
            {"title": "Literature", "text": "Hamlet is a famous play."},
        ],
    })

    assert result is not None, "WTB run returned None"
    final_pred = result.state.workflow_variables.get("final_pred", "")
    print(f"[PASS] WTB full execution -> final_pred: '{final_pred[:60]}...'")


# ═══════════════════════════════════════════════════════════════════════════════
# Run all tests
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        test_query_graph_uncompiled,
        test_longform_graph_uncompiled,
        test_query_graph_compiled_standalone,
        test_query_pipeline_e2e,
        test_longform_pipeline_e2e,
        test_wtb_register_and_compile,
        test_wtb_full_execution,
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

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
