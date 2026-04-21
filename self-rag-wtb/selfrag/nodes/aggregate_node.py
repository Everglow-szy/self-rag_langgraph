"""Aggregate node: picks the best answer across retrieval paths."""
from typing import Any, Dict

from selfrag.constants import control_tokens


def _postprocess(answer: str) -> str:
    """Remove Self-RAG special tokens from generated text."""
    for t in control_tokens:
        answer = answer.replace(t, "")
    return (
        answer.replace("</s>", "")
        .replace("\n", "")
        .replace("<|endoftext|>", "")
    )


def build_aggregate_node():
    """Factory: returns a node that selects the best prediction."""

    def aggregate_node(state):
        results: Dict[str, Dict[str, Any]] = state.get("evidence_results") or {}
        closed = state.get("closed", False)

        if not results:
            raw = state.get("no_retrieval_pred", "")
            pred = _postprocess(raw)
        elif closed:
            answer2score: Dict[str, float] = {}
            for _key, r in results.items():
                ans = _postprocess(r["pred"])
                answer2score[ans] = answer2score.get(ans, 0.0) + r["score"]
            pred = max(answer2score.items(), key=lambda x: x[1])[0]
        else:
            best_key = max(results.items(), key=lambda kv: kv[1]["score"])[0]
            pred = _postprocess(results[best_key]["pred"])

        if pred and pred[0] in ("#", ":"):
            pred = pred[1:]
        return {"final_pred": pred}

    return aggregate_node