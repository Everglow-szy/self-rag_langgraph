"""Init beam node: initializes the prediction tree for beam search."""


def build_init_beam_node():
    """Factory: returns a node that sets up the initial beam tree state."""

    def init_beam_node(state):
        tree = {
            0: {
                "prompt": state["prompt"],
                "pred": "[Retrieval]",
                "processed_pred": "",
                "score": None,
                "ctx": None,
                "parent": None,
            }
        }
        levels = {0: [0]}
        return {
            "prediction_tree": tree,
            "levels": levels,
            "current_depth": 1,
            "node_id_counter": 0,
            "terminated": False,
        }

    return init_beam_node