from __future__ import annotations

from rag_contracts import Generation


def build_node(generation: Generation):
    async def node(state):
        query = state["query"]
        context = state.get("retrieval_results", [])
        result = generation.generate(query=query, context=context, instruction="")
        return {"generation_result": result}

    return node
