from .chunk_node import build_chunk_node
from .embedding_node import build_embedding_node
from .retrieval_node import build_retrieval_node
from .prompt_node import build_prompt_node
from .aggregate_node import build_aggregate_node
from .init_beam_node import build_init_beam_node
from .assemble_node import build_assemble_node

# Nodes requiring vllm are imported lazily to avoid ImportError on machines
# without GPU / vllm installed (e.g. local dev, WTB test harness).
# Import them explicitly when needed:
#   from selfrag.nodes.decision_node import build_decision_node
#   from selfrag.nodes.generation_node import build_evidence_generation_node
#   from selfrag.nodes.longform_decision_node import build_longform_decision_node
#   from selfrag.nodes.longform_no_retrieval_node import build_longform_no_retrieval_node
#   from selfrag.nodes.beam_step_node import build_beam_step_node