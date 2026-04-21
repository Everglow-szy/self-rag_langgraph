"""Self-RAG LangGraph configuration."""
from dataclasses import dataclass


@dataclass
class SelfRAGConfig:
    # ---- Model ----
    model_name: str = "selfrag/selfrag_llama2_7b"
    download_dir: str = "/data1/ragworkspace/self-rag/model_cache"
    dtype: str = "half"

    # ---- Retriever (Contriever) ----
    retriever_model: str = "facebook/contriever-msmarco"
    retriever_device: str = "cpu"

    # ---- Index / Store ----
    store_dir: str = "./out"

    # ---- Retrieval ----
    ndocs: int = 5
    mode: str = "adaptive_retrieval"   # adaptive_retrieval | always_retrieve | no_retrieval
    threshold: float = 0.2

    # ---- Generation ----
    max_new_tokens: int = 100

    # ---- Beam search (long-form) ----
    beam_width: int = 2
    max_depth: int = 7
    ignore_cont: bool = False

    # ---- Scoring weights ----
    w_rel: float = 1.0
    w_sup: float = 1.0
    w_use: float = 0.5
    use_seqscore: bool = False
    closed: bool = False