# Self-RAG WTB

Self-RAG LangGraph 节点的 WTB (Workflow Test Bench) 适配版本。

基于 [self-rag-langgraph](../self-rag-langgraph/) 项目，修改以下内容使其兼容 WTB 框架：

## 与原版的差异

| 改动 | 原版 (self-rag-langgraph) | WTB 适配版 |
|------|--------------------------|-----------|
| 包名 | `langgraph_nodes` (依赖 `sys.path`) | `selfrag` (标准 Python 包) |
| import 路径 | `from utils import PROMPT_DICT` | `from selfrag.constants import PROMPT_DICT` |
| graph builder 返回值 | `g.compile()` (已编译) | `g` (未编译 StateGraph) |
| graph_factory 签名 | 需要 model/tokens/config 参数 | 无参闭包 (通过 `wtb_integration.py` 包装) |
| 节点返回值 | `{**state, "key": val}` (全量拷贝) | `{"key": val}` (仅更新字段) |

## 目录结构

```
self-rag-wtb/
├── README.md
├── wtb_integration.py          # WTB WorkflowProject 注册入口
└── selfrag/
    ├── __init__.py
    ├── config.py               # SelfRAGConfig dataclass
    ├── constants.py            # PROMPT_DICT, TASK_INST, control_tokens (from utils.py)
    ├── state.py                # IndexState / QueryState / LongFormQueryState
    ├── store/
    │   ├── doc_store.py        # JSON-persisted passage store
    │   └── vector_store.py     # numpy cosine-similarity top-k
    ├── nodes/
    │   ├── chunk_node.py
    │   ├── embedding_node.py
    │   ├── retrieval_node.py
    │   ├── prompt_node.py
    │   ├── decision_node.py
    │   ├── generation_node.py
    │   ├── aggregate_node.py
    │   ├── longform_prompt_node.py
    │   ├── longform_decision_node.py
    │   ├── longform_no_retrieval_node.py
    │   ├── init_beam_node.py
    │   ├── beam_step_node.py
    │   └── assemble_node.py
    ├── graph_index.py          # Indexing pipeline (chunk -> embed -> END)
    ├── graph_query.py          # Short-form query pipeline (PopQA)
    └── graph_query_longform.py # Long-form beam search pipeline (ASQA)
```

## WTB 集成用法

```python
from vllm import LLM
from transformers import AutoTokenizer
from wtb.sdk import WTBTestBench
from wtb_integration import create_selfrag_query_project
from selfrag import SelfRAGConfig

# 1. 加载模型
model = LLM("selfrag/selfrag_llama2_7b", dtype="half")
tokenizer = AutoTokenizer.from_pretrained("selfrag/selfrag_llama2_7b")
config = SelfRAGConfig(mode="always_retrieve", ndocs=5)

# 2. 创建 WTB 项目
project = create_selfrag_query_project(model, tokenizer, config)

# 3. 注册并运行
bench = WTBTestBench.create(mode="development")
bench.register_project(project)

result = bench.run("selfrag_query", initial_state={
    "question": "Who wrote Hamlet?",
    "evidences": [{"title": "...", "text": "..."}, ...],
})
print(result.state.workflow_variables["final_pred"])
```

## 独立运行 (不依赖 WTB)

graph builder 接受可选的 `checkpointer` 参数，传入后返回已编译的 graph：

```python
from langgraph.checkpoint.memory import MemorySaver
from selfrag import build_query_graph, SelfRAGConfig

graph = build_query_graph(
    llm_model, ret_tokens, rel_tokens, grd_tokens, ut_tokens,
    config=SelfRAGConfig(),
    checkpointer=MemorySaver(),  # 传入 checkpointer -> 返回已编译 graph
)
result = graph.invoke({"question": "Who wrote Hamlet?", "evidences": [...]})
```