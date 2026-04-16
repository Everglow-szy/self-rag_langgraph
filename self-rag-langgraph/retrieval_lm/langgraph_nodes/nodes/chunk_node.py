"""Chunk node: splits a document into passages and stores them in DocStore."""
import hashlib


def build_chunk_node(doc_store):
    """Factory: returns a node function that chunks a document and saves to DocStore.

    Self-RAG's Wikipedia passages are already paragraph-level, so each passage
    maps directly to one chunk.  For raw text input the node can be extended
    with token-level splitting later.
    """

    def chunk_node(state):
        doc_id = state["doc_id"]
        title = state.get("title", "")
        text = state.get("text", "")

        chunk_id = hashlib.md5(f"{title}||{text}".encode()).hexdigest()
        doc_store.upsert(chunk_id, title=title, text=text, doc_id=doc_id)
        chunks = [{"chunk_id": chunk_id, "title": title, "text": text, "doc_id": doc_id}]
        return {**state, "chunks": chunks}

    return chunk_node
