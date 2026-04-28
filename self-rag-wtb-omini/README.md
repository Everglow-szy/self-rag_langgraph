# self-rag-wtb-eval

Self-RAG unified evaluation framework, bridging Self-RAG's vLLM inference with WTB (Workflow Test Bench) and OminiRAG benchmark adapters through the `rag_contracts` protocol layer.

## Architecture

```
                       rag_contracts (protocol layer)
                     ┌──────────────────────────────┐
                     │  Generation  Retrieval        │
                     │  Reranking   Query             │
                     │  GenerationResult              │
                     └──────┬───────────┬────────────┘
                            │           │
              ┌─────────────┘           └──────────────┐
              ▼                                        ▼
   ┌─────────────────────┐                ┌────────────────────────┐
   │   selfrag/adapters  │                │  OminiRAG benchmark    │
   │                     │                │  adapters              │
   │  SelfRAGGeneration  │◄──────────────►│  HotpotQAAdapter       │
   │  SelfRAGReranking   │  same protocol │  UltraDomainAdapter    │
   │  SelfRAGRetrieval   │                └────────────────────────┘
   └─────────┬───────────┘
             │
   ┌─────────▼───────────┐     ┌───────────────────┐
   │  modular_pipeline   │────►│  WTB TestBench     │
   │  (LangGraph)        │     │  bench.run()       │
   │  query → retrieval  │     └───────────────────┘
   │  → reranking        │
   │  → generation       │
   └─────────────────────┘
```

**Core idea**: `selfrag/adapters.py` wraps Self-RAG's vLLM inference (reflection tokens, logprob scoring) into standard `rag_contracts` protocols. Any framework that speaks `rag_contracts` -- OminiRAG, WTB, or future systems -- can directly use Self-RAG components without modification.

## Project Structure

```
self-rag-wtb-eval/
├── selfrag/
│   ├── adapters.py              # Bidirectional adapters (Self-RAG <-> rag_contracts)
│   ├── modular_pipeline.py      # 4-node LangGraph pipeline (query→retrieval→reranking→generation)
│   ├── state.py                 # TypedDict state schemas (QueryState, SelfRAGModularState, ...)
│   ├── config.py                # SelfRAGConfig dataclass
│   ├── constants.py             # Control tokens, prompt templates, load_special_tokens()
│   ├── graph_query.py           # Original Self-RAG query pipeline (vLLM native)
│   ├── graph_query_longform.py  # Beam-search long-form pipeline
│   ├── graph_index.py           # Chunking + embedding indexing pipeline
│   ├── store/                   # DocStore / VectorStore implementations
│   └── nodes/                   # All graph nodes (native + modular)
│       ├── generation_node.py         # rag_contracts Generation node
│       ├── modular_generation_node.py # Alternative generation node (with final_pred)
│       ├── modular_retrieval_node.py  # rag_contracts Retrieval node
│       ├── modular_reranking_node.py  # rag_contracts Reranking node
│       ├── modular_query_node.py      # rag_contracts Query node
│       ├── decision_node.py           # Retrieval decision ([Retrieval]/[No Retrieval])
│       ├── aggregate_node.py          # Multi-evidence aggregation
│       ├── beam_step_node.py          # Beam search step
│       ├── assemble_node.py           # Beam tree traversal + citation assembly
│       └── ...
├── wtb_integration.py           # WTB WorkflowProject factories
├── run_eval.py                  # Unified evaluation via WTB (PopQA, HotPotQA, UltraDomain, ...)
├── run_eval_omini.py            # Evaluation via OminiRAG benchmark adapters
├── run_hotpotqa_eval.py         # Standalone HotPotQA evaluation
├── prepare_hotpotqa.py          # Data preprocessing: HotPotQA + Contriever retrieval
├── prepare_ultradomain.py       # Data preprocessing: UltraDomain chunking + Contriever retrieval
├── test_wtb_integration.py      # WTB integration tests
└── test_modular_swap.py         # Cross-framework component swap tests
```

## Supported Datasets

| Dataset | Task | Metrics | Data Source |
|---|---|---|---|
| PopQA | Short-form QA | Accuracy (match) | Self-RAG original |
| ASQA | Long-form QA | str_em, ROUGE-L, MAUVE, citation_rec, citation_prec | ALCE benchmark |
| HotPotQA | Multi-hop QA | Exact Match, Token F1 | HuggingFace `hotpotqa/hotpot_qa` |
| UltraDomain | Domain QA | Token F1, Comprehensiveness, Diversity, Empowerment | `TommyChien/UltraDomain` |

## Quick Start

### 1. Install Dependencies

```bash
pip install vllm transformers langgraph rag_contracts
```

### 2. Data Preprocessing

Both HotPotQA and UltraDomain require Contriever retrieval preprocessing before evaluation.

**HotPotQA**:
```bash
python prepare_hotpotqa.py \
    --split validation \
    --output data/hotpotqa_val_contriever.jsonl \
    --num_samples 500 \
    --device cuda
```

**UltraDomain**:
```bash
python prepare_ultradomain.py \
    --data_dir /path/to/UltraDomain/jsonl_files \
    --output data/ultradomain_contriever.jsonl \
    --domains physics cs mathematics \
    --num_samples 200 \
    --device cuda
```

Both scripts produce a unified JSONL format:
```json
{
    "id": "...",
    "question": "...",
    "answer": "...",
    "evidences": [
        {"title": "...", "text": "...", "score": 0.85, "rank": 0},
        ...
    ]
}
```

### 3. Evaluation

Three evaluation entry points are available, each suited for different scenarios:

#### Option A: Via OminiRAG Adapters (`run_eval_omini.py`)

Uses OminiRAG's `HotpotQABenchmarkAdapter` / `UltraDomainBenchmarkAdapter` for evaluation.
Self-RAG components are exposed through `rag_contracts` adapters -- no OminiRAG code changes needed.

```bash
# HotPotQA - generation mode (SelfRAGGeneration adapter directly)
python run_eval_omini.py \
    --dataset hotpotqa \
    --input data/hotpotqa_val_contriever.jsonl \
    --output results/hotpotqa_omini.json \
    --model_name selfrag/selfrag_llama2_7b \
    --eval_mode generation

# HotPotQA - pipeline mode (full LangGraph modular pipeline)
python run_eval_omini.py \
    --dataset hotpotqa \
    --input data/hotpotqa_val_contriever.jsonl \
    --output results/hotpotqa_omini_pipe.json \
    --model_name selfrag/selfrag_llama2_7b \
    --eval_mode pipeline

# UltraDomain - with LLM-as-judge scoring
python run_eval_omini.py \
    --dataset ultradomain \
    --input data/ultradomain_contriever.jsonl \
    --output results/ultradomain_omini.json \
    --model_name selfrag/selfrag_llama2_7b \
    --eval_mode generation \
    --judge_model gpt-4o-mini

# UltraDomain - from OminiRAG KG sample data (no prepare step needed)
python run_eval_omini.py \
    --dataset ultradomain \
    --sample_dir /path/to/OminiRAG/benchmark/sample_data/ultradomain_kg_sample \
    --output results/ultradomain_omini.json \
    --model_name selfrag/selfrag_llama2_7b \
    --eval_mode generation
```

#### Option B: Via WTB (`run_eval.py`)

Uses WTB's `bench.run()` execution controller. Provides checkpointing, state persistence, and execution tracking.

```bash
python run_eval.py \
    --dataset hotpotqa \
    --input data/hotpotqa_val_contriever.jsonl \
    --output results/hotpotqa_wtb.jsonl \
    --model_name selfrag/selfrag_llama2_7b \
    --mode always_retrieve --ndocs 10
```

#### Option C: Standalone (`run_hotpotqa_eval.py`)

Direct LangGraph graph invocation, no external framework dependency.

```bash
python run_hotpotqa_eval.py \
    --input data/hotpotqa_val_contriever.jsonl \
    --output results/hotpotqa_standalone.jsonl \
    --model_name selfrag/selfrag_llama2_7b \
    --mode always_retrieve --ndocs 10
```

## How Self-RAG Connects to HotPotQA and UltraDomain

Self-RAG was originally evaluated only on PopQA, ASQA, and FactScore. Extending it to new datasets like HotPotQA and UltraDomain requires solving two problems: **data format adaptation** and **interface bridging**.

### Data Format Adaptation

HotPotQA and UltraDomain have different input formats from Self-RAG's expected structure. The `prepare_*.py` scripts handle this:

- **HotPotQA**: Each example has 10 context paragraphs (distractor setting). `prepare_hotpotqa.py` encodes them with Contriever and ranks by cosine similarity, producing the `{question, answer, evidences}` format that Self-RAG expects.

- **UltraDomain**: Each example has a book-length context (~80K tokens). `prepare_ultradomain.py` chunks the context into 512-word windows with 50-word overlap, encodes chunks with Contriever, and retrieves the top-K most relevant chunks.

Both scripts produce the same unified JSONL schema, making all downstream evaluation code dataset-agnostic.

### Interface Bridging via rag_contracts

The key challenge is that Self-RAG's internal inference is tightly coupled to vLLM:
- It generates per-passage answers with `SamplingParams(logprobs=...)`.
- It scores each answer using reflection token logprobs (ISREL, ISSUP, ISUSE).
- It selects the best answer based on a weighted composite score.

The `selfrag/adapters.py` module wraps this logic into standard `rag_contracts` protocols:

| Adapter | Protocol | What it wraps |
|---|---|---|
| `SelfRAGGeneration` | `Generation.generate(query, context) -> GenerationResult` | Per-passage vLLM generation + logprob scoring + best-answer selection |
| `SelfRAGReranking` | `Reranking.rerank(query, results) -> list[RetrievalResult]` | Evidence-generation scoring (generates to score, caches predictions) |
| `SelfRAGRetrieval` | `Retrieval.retrieve(queries) -> list[RetrievalResult]` | Contriever encode + VectorStore cosine search |

Once wrapped, Self-RAG components can be used by any framework that speaks `rag_contracts`:

```python
from selfrag.adapters import SelfRAGGeneration
from benchmark.hotpotqa_adapter import HotpotQABenchmarkAdapter

# Self-RAG generation satisfies rag_contracts.Generation protocol
generation = SelfRAGGeneration(model=llm, rel_tokens=..., grd_tokens=..., ut_tokens=...)

# OminiRAG adapter accepts any rag_contracts.Generation -- zero code changes
adapter = HotpotQABenchmarkAdapter()
results = adapter.evaluate_generation(data, generation)
```

For full pipeline mode, the modular pipeline (`build_selfrag_modular_graph`) constructs a LangGraph StateGraph with state schema `SelfRAGModularState`. This state includes `query`, `generation_result`, `query_id`, `answers`, and `test_data_name` fields -- exactly what OminiRAG's `evaluate_pipeline()` expects:

```python
from selfrag.modular_pipeline import build_selfrag_modular_graph

graph = build_selfrag_modular_graph(
    retrieval=my_retrieval,
    generation=SelfRAGGeneration(...),
    reranking=SelfRAGReranking(...),
)

# OminiRAG calls graph.ainvoke({"query": ...}) and reads state["generation_result"]
results = adapter.evaluate_pipeline(data, graph)
```

### Evaluation Flow Diagram

```
 prepare_hotpotqa.py                    run_eval_omini.py
 prepare_ultradomain.py
 ┌──────────────────┐          ┌──────────────────────────────────┐
 │  Raw Dataset      │          │                                  │
 │  (HuggingFace /   │──JSONL──►│  Load data                       │
 │   local files)    │          │       │                          │
 └──────────────────┘          │       ▼                          │
                               │  SelfRAGGeneration (adapter)     │
                               │       │                          │
                               │       ▼                          │
                               │  OminiRAG BenchmarkAdapter       │
                               │  .evaluate_generation(data, gen) │
                               │       │                          │
                               │       ▼                          │
                               │  EM / F1 / LLM-judge metrics    │
                               └──────────────────────────────────┘
```

## Self-RAG Scoring Parameters

| Parameter | Default | Description |
|---|---|---|
| `--w_rel` | 1.0 | Weight for relevance score (ISREL) |
| `--w_sup` | 1.0 | Weight for grounding score (ISSUP) |
| `--w_use` | 0.5 | Weight for utility score (ISUSE) |
| `--use_seqscore` | False | Include sequence log-probability in final score |
| `--max_new_tokens` | 100 | Max tokens per generation |
| `--ndocs` | 10 | Number of retrieved passages |

For PopQA paper reproduction: `--ndocs 10 --use_seqscore --w_use 1.0`.

## Cross-Framework Component Swap

The `rag_contracts` protocol layer enables swapping components across RAG frameworks:

```python
from selfrag.adapters import SelfRAGGeneration, SelfRAGReranking
from selfrag.modular_pipeline import build_selfrag_modular_graph

# Use Self-RAG generation with an external retrieval system
from some_other_rag import CustomRetrieval

graph = build_selfrag_modular_graph(
    retrieval=CustomRetrieval(...),       # From another framework
    generation=SelfRAGGeneration(...),    # Self-RAG's vLLM inference
    reranking=SelfRAGReranking(...),      # Self-RAG's logprob-based reranking
)
```

This is the same mechanism that allows OminiRAG and WTB to use Self-RAG components without any code modifications to either framework.
