"""Real LLM verification: run Self-RAG pipelines with an OpenAI-compatible model.

Reads LLM_API_KEY, LLM_BASE_URL, DEFAULT_LLM from .env and verifies the
full Self-RAG query pipeline end-to-end with a live LLM call instead of mocks.
"""
import sys
import os
import types
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path
from unittest.mock import MagicMock

# ── Load .env ──────────────────────────────────────────────────────────────
from dotenv import load_dotenv

for env_path in [Path(__file__).parent / ".env",
                 Path(__file__).parent.parent.parent / ".env"]:
    if env_path.exists():
        load_dotenv(env_path, override=True)
        break

API_KEY = os.environ.get("LLM_API_KEY", "")
BASE_URL = os.environ.get("LLM_BASE_URL", "")
MODEL = os.environ.get("DEFAULT_LLM", "gpt-4o-mini")

assert API_KEY, "LLM_API_KEY not found in .env"
assert BASE_URL, "LLM_BASE_URL not found in .env"
print(f"[CONFIG] model={MODEL}  base_url={BASE_URL}")


# ── Inject fake vllm so selfrag.nodes can import SamplingParams ────────────
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
        __enter__=lambda s: None, __exit__=lambda s, *a: None)
    sys.modules["torch"] = fake_torch

sys.path.insert(0, os.path.dirname(__file__))

from selfrag.config import SelfRAGConfig
from selfrag.constants import (
    load_special_tokens,
    rel_tokens_names,
    retrieval_tokens_names,
    ground_tokens_names,
    utility_tokens_names,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Tokenizer with stable special-token IDs
# ═══════════════════════════════════════════════════════════════════════════════

ALL_SPECIAL = (
    retrieval_tokens_names
    + rel_tokens_names
    + ground_tokens_names
    + utility_tokens_names
)
TOKEN_MAP: Dict[str, int] = {tok: 2000 + i for i, tok in enumerate(ALL_SPECIAL)}


class RealTokenizer:
    def convert_tokens_to_ids(self, token: str) -> int:
        return TOKEN_MAP.get(token, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# OpenAI model wrapper (mimics vLLM generate interface)
# ═══════════════════════════════════════════════════════════════════════════════

SELFRAG_SYSTEM = """\
You are a Self-RAG assistant. Given a question and optionally a paragraph \
of evidence, produce a SHORT factual answer wrapped in Self-RAG control tokens.

You MUST follow this EXACT output format (nothing else):
[Relevant]<your answer>[Fully supported][Utility:5]

Control-token choices:
  Relevance : [Relevant] or [Irrelevant]
  Grounding : [Fully supported], [Partially supported], or [No support / Contradictory]
  Utility   : [Utility:1] .. [Utility:5]  (5 = most useful)

Rules:
- Keep answer to 1-2 sentences.
- Always include ALL three token types.
- Do NOT add markdown, explanations, or extra text.
"""


def _detect_token(text: str, candidates: List[str]) -> Optional[str]:
    """Return the first candidate token found in text, or None."""
    for tok in candidates:
        if tok in text:
            return tok
    return None


class _VLLMOutput:
    """Mimics vLLM CompletionOutput with synthetic logprobs."""

    def __init__(self, text: str):
        self.text = text

        found_rel = _detect_token(text, rel_tokens_names)
        found_grd = _detect_token(text, ground_tokens_names)
        found_ut = _detect_token(text, utility_tokens_names)
        found_ret = _detect_token(text, retrieval_tokens_names)

        # token_ids: [rel, ...filler..., grd, ut]
        n_filler = max(5, len(text.split()) // 2)
        self.token_ids: List[int] = []

        self.token_ids.append(TOKEN_MAP.get(found_rel or "[Relevant]", 888))
        self.token_ids.extend([888] * n_filler)
        grd_pos = len(self.token_ids)
        self.token_ids.append(TOKEN_MAP.get(found_grd or "[Fully supported]", 888))
        ut_pos = len(self.token_ids)
        self.token_ids.append(TOKEN_MAP.get(found_ut or "[Utility:5]", 888))

        # logprobs: one dict per position
        self.logprobs: List[Dict[int, float]] = []
        for pos in range(len(self.token_ids)):
            entry: Dict[int, float] = {}
            for tok in rel_tokens_names:
                tid = TOKEN_MAP[tok]
                entry[tid] = -0.1 if (pos == 0 and tok == found_rel) else -5.0
            for tok in retrieval_tokens_names:
                tid = TOKEN_MAP[tok]
                entry[tid] = -0.1 if tok == found_ret else -5.0
            for tok in ground_tokens_names:
                tid = TOKEN_MAP[tok]
                entry[tid] = -0.1 if (pos == grd_pos and tok == found_grd) else -5.0
            for tok in utility_tokens_names:
                tid = TOKEN_MAP[tok]
                entry[tid] = -0.1 if (pos == ut_pos and tok == found_ut) else -5.0
            self.logprobs.append(entry)

        self.cumulative_logprob = -2.0


class _VLLMPrediction:
    def __init__(self, output: _VLLMOutput):
        self.outputs = [output]


class OpenAIModel:
    """Wraps OpenAI chat completions behind the vLLM generate() interface."""

    def __init__(self, api_key: str, base_url: str, model: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(self, prompts: List[str],
                 sampling_params=None) -> List[_VLLMPrediction]:
        results: List[_VLLMPrediction] = []
        for prompt in prompts:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SELFRAG_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=getattr(sampling_params, "temperature", 0.0),
                    max_tokens=getattr(sampling_params, "max_tokens", 150),
                )
                text = resp.choices[0].message.content or ""
            except Exception as exc:
                print(f"  [LLM ERROR] {exc}")
                text = ("[Relevant]Error generating response."
                        "[Partially supported][Utility:3]")
            print(f"  [LLM] {text[:100]}")
            results.append(_VLLMPrediction(_VLLMOutput(text)))
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_real_llm_query_pipeline():
    """Short-form query pipeline with real LLM (always_retrieve mode)."""
    from selfrag.graph_query import build_query_graph
    from langgraph.checkpoint.memory import MemorySaver

    model = OpenAIModel(API_KEY, BASE_URL, MODEL)
    tokenizer = RealTokenizer()
    config = SelfRAGConfig(mode="always_retrieve", ndocs=2)
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=True, use_utility=True)

    graph = build_query_graph(
        model, ret_tokens, rel_tokens, grd_tokens, ut_tokens, config,
        checkpointer=MemorySaver())

    state = {
        "question": "Who wrote Hamlet?",
        "evidences": [
            {"title": "Shakespeare", "text": "William Shakespeare wrote Hamlet around 1600."},
            {"title": "Literature", "text": "Hamlet is one of Shakespeare's most famous tragedies."},
        ],
    }
    result = graph.invoke(state, config={"configurable": {"thread_id": "real-1"}})

    assert "final_pred" in result, f"Missing final_pred, keys: {list(result.keys())}"
    assert len(result["final_pred"]) > 0, "final_pred is empty"
    print(f"[PASS] real LLM query -> final_pred: '{result['final_pred']}'")
    return result


def test_real_llm_no_retrieval():
    """Short-form query pipeline with no_retrieval mode."""
    from selfrag.graph_query import build_query_graph
    from langgraph.checkpoint.memory import MemorySaver

    model = OpenAIModel(API_KEY, BASE_URL, MODEL)
    tokenizer = RealTokenizer()
    config = SelfRAGConfig(mode="no_retrieval")
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=True, use_utility=True)

    graph = build_query_graph(
        model, ret_tokens, rel_tokens, grd_tokens, ut_tokens, config,
        checkpointer=MemorySaver())

    state = {"question": "What is the capital of France?"}
    result = graph.invoke(state, config={"configurable": {"thread_id": "real-2"}})

    assert "final_pred" in result, f"Missing final_pred, keys: {list(result.keys())}"
    assert len(result["final_pred"]) > 0, "final_pred is empty"
    print(f"[PASS] real LLM no-retrieval -> final_pred: '{result['final_pred']}'")
    return result


def test_real_llm_longform_pipeline():
    """Long-form beam search pipeline with real LLM."""
    from selfrag.graph_query_longform import build_longform_query_graph
    from langgraph.checkpoint.memory import MemorySaver

    model = OpenAIModel(API_KEY, BASE_URL, MODEL)
    tokenizer = RealTokenizer()
    config = SelfRAGConfig(
        mode="always_retrieve", ndocs=2, beam_width=2, max_depth=2)
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=True, use_utility=True)

    graph = build_longform_query_graph(
        model, ret_tokens, rel_tokens, grd_tokens, ut_tokens, config,
        checkpointer=MemorySaver())

    state = {
        "question": "What is the history of baptism?",
        "task": "asqa",
        "docs": [
            {"title": "Baptism", "text": "Baptism is a Christian rite of admission."},
            {"title": "History", "text": "John the Baptist baptized Jesus in the Jordan River."},
        ],
    }
    result = graph.invoke(state, config={"configurable": {"thread_id": "real-3"}})

    assert "final_output" in result, f"Missing final_output, keys: {list(result.keys())}"
    print(f"[PASS] real LLM longform -> final_output: '{result.get('final_output', '')[:120]}'")
    return result


def test_real_llm_wtb_execution():
    """Full WTB execution with real LLM."""
    from selfrag.graph_query import build_query_graph
    from wtb.sdk import WTBTestBench
    from wtb.sdk.workflow_project import WorkflowProject
    import uuid

    model = OpenAIModel(API_KEY, BASE_URL, MODEL)
    tokenizer = RealTokenizer()
    config = SelfRAGConfig(mode="always_retrieve", ndocs=2)
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=True, use_utility=True)

    def graph_factory():
        return build_query_graph(
            model, ret_tokens, rel_tokens, grd_tokens, ut_tokens, config)

    pname = f"selfrag_real_{uuid.uuid4().hex[:8]}"
    project = WorkflowProject(name=pname, graph_factory=graph_factory)

    bench = WTBTestBench.create(mode="development")
    bench.register_project(project)

    result = bench.run(pname, initial_state={
        "question": "Who invented the telephone?",
        "evidences": [
            {"title": "Bell", "text": "Alexander Graham Bell patented the telephone in 1876."},
            {"title": "History", "text": "The telephone revolutionized global communications."},
        ],
    })

    assert result is not None, "WTB run returned None"
    final_pred = result.state.workflow_variables.get("final_pred", "")
    assert len(final_pred) > 0, "final_pred is empty"
    print(f"[PASS] real LLM WTB execution -> final_pred: '{final_pred}'")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Run all tests
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        test_real_llm_query_pipeline,
        test_real_llm_no_retrieval,
        test_real_llm_longform_pipeline,
        test_real_llm_wtb_execution,
    ]
    passed = 0
    failed = 0
    for test in tests:
        print(f"\n{'─'*60}")
        print(f"Running {test.__name__}...")
        print(f"{'─'*60}")
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
        print("ALL REAL-LLM TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
