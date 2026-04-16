# Self-RAG LangGraph

A pluggable **LangGraph** refactor of [Self-RAG](https://github.com/AkariAsai/self-rag) (Asai et al., 2023), decomposing the monolithic `call_model_rerank_w_scores_batch` / `run_step_generation_batch` functions into reusable graph nodes with explicit **Index / Retrieval / Store** layers.

The LangGraph pipeline is numerically equivalent to the original Self-RAG implementation (verified at **100% agreement** on both PopQA and ASQA), while exposing each stage (prompting, adaptive retrieval decision, evidence generation, beam search, aggregation) as an independently testable node.

---

## Architecture

```
retrieval_lm/langgraph_nodes/
├── config.py                    # SelfRAGConfig dataclass
├── state.py                     # IndexState / QueryState / LongFormQueryState
├── store/
│   ├── doc_store.py             # JSON-persisted passage store
│   └── vector_store.py          # numpy cosine-similarity top-k
├── nodes/
│   ├── chunk_node.py            # passage chunking
│   ├── embedding_node.py        # Contriever mean-pool encoding
│   ├── retrieval_node.py        # top-k retrieval (or pass-through when ctxs given)
│   ├── prompt_node.py           # Self-RAG instruction prompt
│   ├── decision_node.py         # [Retrieval] vs [No Retrieval] logprob decision
│   ├── generation_node.py       # evidence generation + ISREL/ISSUP/ISUSE scoring
│   ├── aggregate_node.py        # pick best (open-form) / group-by-answer (closed)
│   ├── longform_prompt_node.py
│   ├── longform_decision_node.py
│   ├── longform_no_retrieval_node.py
│   ├── init_beam_node.py        # seed prediction tree
│   ├── beam_step_node.py        # one beam-search expansion step
│   └── assemble_node.py         # tree traversal → final answer + citations
├── graph_index.py               # Indexing pipeline
├── graph_query.py               # Short-form query pipeline (PopQA)
└── graph_query_longform.py      # Long-form tree-beam pipeline (ASQA)
```

### Pipelines

**1. Indexing pipeline** (`graph_index.py`)

```
START → chunk → embedding → END
```

Chunks passages into `DocStore`, encodes with `facebook/contriever-msmarco`, and persists vectors to `VectorStore`.

**2. Short-form query pipeline** (`graph_query.py`) — PopQA

```
START → prepare_prompt → retrieval_decision ─┬→ retrieve → evidence_generation → aggregate → END
                                              └→ no_retrieval_generation ────────→ aggregate → END
```

Single-step generation with per-passage scoring; supports `adaptive_retrieval`, `always_retrieve`, and `no_retrieval` modes.

**3. Long-form query pipeline** (`graph_query_longform.py`) — ASQA / ELI5

```
START → prepare_prompt → retrieval_decision ─┬→ init_beam ─┐
                                              │             ↓
                                              │         beam_step ─→ assemble → END
                                              │             ↑__cycle__|
                                              └→ no_retrieval_generation → END
```

Tree-based beam search over `[Retrieval]` segments, collecting citations via `[Continue]` / `[No Retrieval]` terminators.

---

## Folder Layout

```
self-rag-langgraph/
├── README.md
├── requirements.txt
├── .gitignore
├── retrieval_lm/
│   ├── run_popqa.py                     # entry: short-form comparison
│   ├── run_asqa.py                      # entry: long-form comparison
│   ├── run_compare_eval.py              # PopQA comparison driver
│   ├── run_compare_eval_longform.py     # ASQA comparison driver
│   ├── run_short_form.py                # original baseline (short-form)
│   ├── run_long_form_static.py          # original baseline (long-form)
│   ├── utils.py                         # PROMPT_DICT, control tokens, helpers
│   ├── metrics.py                       # match / accuracy / f1
│   └── langgraph_nodes/                 # (see Architecture above)
└── eval_results/
    ├── compare_popqa.json               # PopQA per-sample comparison
    ├── orig_asqa_output.json.score      # ASQA ALCE metrics (original)
    └── lg_asqa_output.json.score        # ASQA ALCE metrics (LangGraph)
```

---

## Installation

```bash
# Python 3.10, CUDA 12.1, one 24 GB GPU (RTX 4090 class) is enough for Self-RAG 7B (fp16)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The Self-RAG 7B checkpoint is auto-downloaded by vLLM from `selfrag/selfrag_llama2_7b`. For users in regions with restricted HF access:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
```

---

## Usage

### Short-form (PopQA long-tail, 1,399 samples)

```bash
cd retrieval_lm
python run_popqa.py                                   # full run, adaptive retrieval, top-5
python run_popqa.py --num_samples 100                 # smoke test
python run_popqa.py --mode always_retrieve --ndocs 10 # ablation
```

### Long-form (ALCE-ASQA, 948 samples)

```bash
cd retrieval_lm
python run_asqa.py                                    # full run
python run_asqa.py --num_samples 50 --beam_width 2    # smoke test

# The input file ALCE-main/data/asqa_eval_gtr_top100.json comes from
# https://github.com/princeton-nlp/ALCE
```

Both drivers run the **original** `run_short_form.py` / `run_long_form_static.py` path and the **LangGraph** path on identical inputs, then emit a per-sample agreement report and ALCE-style scores.

### Programmatic use

```python
from langgraph_nodes import SelfRAGConfig, build_query_graph

cfg = SelfRAGConfig(
    model_name="selfrag/selfrag_llama2_7b",
    mode="adaptive_retrieval",
    ndocs=5,
    threshold=0.2,
    w_rel=1.0, w_sup=1.0, w_use=0.5,
)
graph = build_query_graph(cfg)

state = {
    "question": "Who wrote Hamlet?",
    "evidences": [{"title": "...", "text": "..."}, ...],  # or run index pipeline first
}
out = graph.invoke(state)
print(out["final_answer"])
```

---

## Experiment Results

All numbers are from identical prompts, retriever, and checkpoint on one RTX 4090 D.

### PopQA long-tail (1,399 samples, adaptive retrieval, top-5)

| Pipeline            | Match |
|---------------------|------:|
| Original Self-RAG   | 52.47 |
| LangGraph refactor  | 52.47 |
| Agreement           | **1399 / 1399 (100%)** |

### ALCE-ASQA (948 samples, always-retrieve, beam width 2, depth 7)

| Pipeline            | str_em | QA-EM | QA-F1 |
|---------------------|-------:|------:|------:|
| Original Self-RAG   |  30.28 | 18.52 | 24.01 |
| LangGraph refactor  |  30.28 | 18.52 | 24.01 |
| Agreement           | **948 / 948 (100%)** |

The LangGraph path reproduces the original numbers exactly because each node faithfully ports the scoring formula from `run_short_form.py:call_model_rerank_w_scores_batch` and `run_long_form_static.py:run_step_generation_batch`.

---

## Reused Code

| Source                                              | Reused component                           |
|-----------------------------------------------------|--------------------------------------------|
| `utils.py:PROMPT_DICT`                              | Self-RAG instruction templates             |
| `utils.py:load_special_tokens`                      | reflection token ids                       |
| `utils.py:control_tokens`                           | postprocess cleanup                        |
| `run_short_form.py:call_model_rerank_w_scores_batch`| ISREL / ISSUP / ISUSE scoring formula      |
| `run_long_form_static.py:run_step_generation_batch` | beam step scoring + tree bookkeeping       |
| `run_short_form.py:postprocess_answer_option_conditioned` | answer cleanup                       |

---

## References

- Asai et al., *"Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"*, ICLR 2024. [[paper](https://arxiv.org/abs/2310.11511)] [[repo](https://github.com/AkariAsai/self-rag)]
- Gao et al., *"Enabling Large Language Models to Generate Text with Citations"*, EMNLP 2023. [[ALCE repo](https://github.com/princeton-nlp/ALCE)]
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [facebook/contriever-msmarco](https://huggingface.co/facebook/contriever-msmarco)
