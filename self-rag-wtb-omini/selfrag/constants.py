"""Self-RAG control tokens, prompt templates, and special token utilities.

Inlined from the original Self-RAG codebase to avoid external dependency.
These are the special tokens the Self-RAG model uses for retrieval decisions,
relevance assessment, grounding verification, and utility scoring.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

# ── Retrieval decision tokens ────────────────────────────────────────────

retrieval_tokens_names = [
    "[Retrieval]",
    "[No Retrieval]",
    "[Continue to Use Evidence]",
]

# ── Relevance tokens (ISREL) ─────────────────────────────────────────────

rel_tokens_names = [
    "[Relevant]",
    "[Irrelevant]",
]

# ── Grounding tokens (ISSUP) ─────────────────────────────────────────────

ground_tokens_names = [
    "[Fully supported]",
    "[Partially supported]",
    "[No support / Contradictory]",
]

# ── Utility tokens (ISUSE) ───────────────────────────────────────────────

utility_tokens_names = [
    "[Utility:1]",
    "[Utility:2]",
    "[Utility:3]",
    "[Utility:4]",
    "[Utility:5]",
]

# ── Aggregate list (used by _postprocess to strip tokens from output) ────

control_tokens = (
    retrieval_tokens_names
    + rel_tokens_names
    + ground_tokens_names
    + utility_tokens_names
    + ["<paragraph>", "</paragraph>"]
)

# ── Prompt templates (Self-RAG / Alpaca style) ───────────────────────────

PROMPT_DICT = {
    "prompt_input": (
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:\n"
    ),
    "prompt_no_input": (
        "### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    ),
}


# ── Token ID loader ──────────────────────────────────────────────────────

def load_special_tokens(
    tokenizer: Any,
    use_grounding: bool = True,
    use_utility: bool = True,
) -> Tuple[
    Optional[Dict[str, int]],
    Dict[str, int],
    Optional[Dict[str, int]],
    Optional[Dict[str, int]],
]:
    """Map Self-RAG special token names to integer IDs via the tokenizer.

    Returns ``(ret_tokens, rel_tokens, grd_tokens, ut_tokens)`` where each
    is a ``{token_name: token_id}`` dict (or ``None`` when disabled).
    """
    ret_tokens = {
        t: tokenizer.convert_tokens_to_ids(t)
        for t in retrieval_tokens_names
    }
    rel_tokens = {
        t: tokenizer.convert_tokens_to_ids(t)
        for t in rel_tokens_names
    }

    grd_tokens: Optional[Dict[str, int]] = None
    if use_grounding:
        grd_tokens = {
            t: tokenizer.convert_tokens_to_ids(t)
            for t in ground_tokens_names
        }

    ut_tokens: Optional[Dict[str, int]] = None
    if use_utility:
        ut_tokens = {
            t: tokenizer.convert_tokens_to_ids(t)
            for t in utility_tokens_names
        }

    return ret_tokens, rel_tokens, grd_tokens, ut_tokens
