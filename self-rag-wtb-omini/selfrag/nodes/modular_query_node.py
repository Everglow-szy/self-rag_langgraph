"""Modular query-processing node: accepts a canonical ``rag_contracts.Query`` via DI."""

from __future__ import annotations

from rag_contracts import IdentityQuery, Query, QueryContext


def build_node(query: Query | None = None):
    """Build a query-processing node that uses the canonical Query protocol."""
    _query = query or IdentityQuery()

    async def node(state):
        raw_query = state.get("query", state.get("question", ""))
        context = QueryContext(topic=raw_query)
        expanded = _query.process(raw_query, context)
        return {"expanded_queries": expanded, "query": raw_query}

    return node
