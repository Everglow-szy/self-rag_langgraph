"""Embedding node: encodes passages with Contriever and stores vectors."""
import numpy as np
import torch


def build_embedding_node(vector_store, retriever_tokenizer, retriever_model):
    """Factory: returns a node that embeds chunks and saves to VectorStore."""

    def _encode(texts):
        """Encode a list of texts with Contriever (mean-pool last hidden state)."""
        inputs = retriever_tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        device = next(retriever_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = retriever_model(**inputs)
        # Mean pooling over token embeddings (mask out padding)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        embeddings = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        return embeddings.cpu().numpy()

    def embedding_node(state):
        chunks = state.get("chunks", [])
        if not chunks:
            return state

        ids = [c["chunk_id"] for c in chunks]
        texts = [f"{c['title']}\n{c['text']}" for c in chunks]
        vecs = _encode(texts)
        vector_store.upsert(ids, vecs)
        return state

    return embedding_node
