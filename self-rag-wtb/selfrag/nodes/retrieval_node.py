"""Retrieval node: fetches top-k passages from VectorStore or uses pre-retrieved ctxs."""
import numpy as np
import torch


def build_retrieval_node(config, doc_store, vector_store,
                         retriever_tokenizer=None, retriever_model=None):
    """Factory: returns a node that retrieves passages.

    If pre-retrieved ``evidences`` exist in state, uses those directly
    (compatibility mode for datasets with ctxs). Otherwise, encodes the
    query with Contriever and does cosine top-k against VectorStore.
    """

    def _encode_query(text):
        inputs = retriever_tokenizer(
            [text], padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        device = next(retriever_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = retriever_model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        emb = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        return emb.cpu().numpy()[0]

    def retrieval_node(state):
        ndocs = state.get("ndocs", config.ndocs)
        pre_evidences = state.get("evidences")

        # Compatibility mode: use pre-retrieved ctxs from dataset
        if pre_evidences:
            passages = pre_evidences[:ndocs]
            return {"retrieved_passages": passages}

        # Real retrieval: encode query and search VectorStore
        if retriever_model is None or vector_store is None:
            return {"retrieved_passages": []}

        query_vec = _encode_query(state["question"])
        results = vector_store.search(query_vec, top_k=ndocs)
        passages = []
        for chunk_id, score in results:
            doc = doc_store.get(chunk_id)
            if doc:
                passages.append({
                    "title": doc["title"],
                    "text": doc["text"],
                    "retrieval_score": score,
                })
        return {"retrieved_passages": passages}

    return retrieval_node