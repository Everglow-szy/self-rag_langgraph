"""Assemble node: traverses the beam tree and builds the final output with citations.

Faithfully ports the tree traversal and citation assembly from
``run_long_form_static.py:call_model_beam_batch`` (lines 256-297)
and the ASQA post-processing from ``main()`` (lines 387-435).
"""
from utils import postprocess, fix_spacing


def build_assemble_node():
    """Factory: returns a node that assembles the final long-form output."""

    def assemble_node(state):
        do_retrieve = state.get("do_retrieve", False)

        # --- No retrieval path: just return the simple prediction ---
        if not do_retrieve:
            pred = state.get("no_retrieval_pred", "")
            cleaned = pred.split("\n\n")[0] if pred else ""
            return {
                **state,
                "final_pred": cleaned,
                "final_output": fix_spacing(postprocess(cleaned)),
                "output_docs": [],
                "intermediate": {},
            }

        # --- Beam search path: traverse the tree ---
        tree = state.get("prediction_tree", {})
        levels = state.get("levels", {})
        ignore_cont = state.get("ignore_cont", False)

        # Remove empty levels and level 0
        levels = {k: v for k, v in levels.items() if len(v) > 0 and k != 0}
        if not levels:
            return {
                **state,
                "final_pred": "",
                "final_output": "",
                "output_docs": [],
                "intermediate": {},
            }

        max_level = max(levels.keys())
        best_selections = {}
        for path_i, node in enumerate(levels[max_level]):
            if node == 0:
                break
            best_selections[path_i] = [node]
            current_node = node
            while current_node is not None:
                parent = tree[current_node].get("parent")
                if parent is not None:
                    best_selections[path_i] = [parent] + best_selections[path_i]
                current_node = parent

        # Build final predictions per path
        final_prediction = {}
        splitted_sentences = {}
        original_splitted_sentences = {}
        ctxs_per_path = {}
        for path_i, nodes in best_selections.items():
            def _include(n):
                if n is None:
                    return False
                if ignore_cont and "[No support / Contradictory]" in tree[n].get("processed_pred", ""):
                    return False
                return True

            final_prediction[path_i] = " ".join([
                tree[n]["processed_pred"] for n in nodes if _include(n)
            ])
            splitted_sentences[path_i] = [
                tree[n]["processed_pred"] for n in nodes if _include(n)
            ]
            original_splitted_sentences[path_i] = [
                tree[n]["pred"] for n in nodes if _include(n)
            ]
            ctxs_per_path[path_i] = [
                tree[n].get("ctx") for n in nodes if _include(n)
            ]

        intermediate = {
            "final_prediction": final_prediction,
            "splitted_sentences": splitted_sentences,
            "original_splitted_sentences": original_splitted_sentences,
            "best_selections": best_selections,
            "ctxs": ctxs_per_path,
        }

        # --- ASQA/ELI5 citation assembly (from main() lines 398-435) ---
        task = state.get("task", "asqa")
        if task in ("asqa", "eli5"):
            final_output = ""
            docs = []
            prev_gen = []
            if "splitted_sentences" not in intermediate:
                final_output = postprocess(final_prediction.get(0, ""))
            else:
                # If path 0 is empty, try path 1
                if len(postprocess(final_prediction.get(0, ""))) == 0 and 1 in splitted_sentences:
                    splitted_sentences[0] = splitted_sentences[1]
                    ctxs_per_path[0] = ctxs_per_path[1]
                for idx, (sent, doc) in enumerate(
                    zip(splitted_sentences.get(0, []), ctxs_per_path.get(0, []))
                ):
                    if len(sent) == 0:
                        continue
                    pp = postprocess(sent)
                    if pp in prev_gen:
                        continue
                    prev_gen.append(pp)
                    final_output += pp[:-1] + " [{}]".format(idx) + ". "
                    docs.append(doc)
                if final_output and final_output[-1] == " ":
                    final_output = final_output[:-1]
                final_output = fix_spacing(final_output)
                final_output = final_output.replace(
                    ".[Continue to Use Evidence]", " [1]. ")
                final_output = final_output.replace(". [1] ", " [1]. ")
        elif task == "factscore":
            final_output = fix_spacing(postprocess(final_prediction.get(0, "")))
            docs = []
        else:
            final_output = postprocess(final_prediction.get(0, ""))
            docs = []

        return {
            **state,
            "final_pred": final_prediction.get(0, ""),
            "final_output": final_output,
            "output_docs": docs,
            "intermediate": intermediate,
        }

    return assemble_node
