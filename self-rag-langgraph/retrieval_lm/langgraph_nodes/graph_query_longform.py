"""Long-form query pipeline with tree-based beam search.

Graph structure:
  START → prepare_prompt → retrieval_decision
    → (no retrieve) → no_retrieval_generation → assemble_output → END
    → (retrieve)    → init_beam → beam_step → check_depth
                                    ↑ (loop)       ↓
                                    └──────── (continue)
                                               (done) → assemble_output → END
"""
from langgraph.graph import END, START, StateGraph

from langgraph_nodes.state import LongFormQueryState
from langgraph_nodes.nodes.longform_prompt_node import build_longform_prompt_node
from langgraph_nodes.nodes.longform_decision_node import build_longform_decision_node
from langgraph_nodes.nodes.longform_no_retrieval_node import build_longform_no_retrieval_node
from langgraph_nodes.nodes.init_beam_node import build_init_beam_node
from langgraph_nodes.nodes.beam_step_node import build_beam_step_node
from langgraph_nodes.nodes.assemble_node import build_assemble_node


def _route_after_decision(state):
    return "init_beam" if state.get("do_retrieve") else "no_retrieval_generation"


def _check_depth(state):
    """Decide whether to continue beam search or assemble output."""
    depth = state.get("current_depth", 1)
    max_depth = state.get("max_depth", 7)
    terminated = state.get("terminated", False)
    if terminated or depth > max_depth:
        return "assemble_output"
    return "beam_step"


def build_longform_query_graph(
    llm_model,
    ret_tokens, rel_tokens, grd_tokens, ut_tokens,
    config,
):
    """Construct the long-form query StateGraph with beam search loop.

    Args:
        llm_model: vLLM model.
        ret_tokens / rel_tokens / grd_tokens / ut_tokens: reflection token maps.
        config: SelfRAGConfig instance (should have beam_width, max_depth).

    Returns:
        Compiled LangGraph graph.
    """
    g = StateGraph(LongFormQueryState)

    # ---- Nodes ----
    g.add_node("prepare_prompt", build_longform_prompt_node())
    g.add_node("retrieval_decision", build_longform_decision_node(llm_model, ret_tokens, config))
    g.add_node("no_retrieval_generation", build_longform_no_retrieval_node(llm_model, config))
    g.add_node("init_beam", build_init_beam_node())
    g.add_node("beam_step", build_beam_step_node(
        llm_model, rel_tokens, grd_tokens, ret_tokens, ut_tokens, config))
    g.add_node("assemble_output", build_assemble_node())

    # ---- Edges ----
    g.add_edge(START, "prepare_prompt")
    g.add_edge("prepare_prompt", "retrieval_decision")
    g.add_conditional_edges(
        "retrieval_decision",
        _route_after_decision,
        {
            "init_beam": "init_beam",
            "no_retrieval_generation": "no_retrieval_generation",
        },
    )
    g.add_edge("init_beam", "beam_step")
    g.add_conditional_edges(
        "beam_step",
        _check_depth,
        {
            "beam_step": "beam_step",
            "assemble_output": "assemble_output",
        },
    )
    g.add_edge("no_retrieval_generation", "assemble_output")
    g.add_edge("assemble_output", END)

    return g.compile()
