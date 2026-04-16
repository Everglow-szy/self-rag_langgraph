"""Entry point: run ASQA long-form comparison experiment.

Usage:
    python run_asqa.py                              # default: ALCE eval = qa only
    python run_asqa.py --num_samples 50             # smoke test
    python run_asqa.py --alce_eval full             # + rouge + mauve (1 GPU)
    python run_asqa.py --alce_eval all              # + citation prec/rec (needs 2+ GPUs)
    python run_asqa.py --alce_eval none             # skip ALCE eval
"""
import os
import sys
import json
import subprocess
import argparse
from datetime import datetime


# ---- ALCE eval flag sets ------------------------------------------------
# Keys are values of --alce_eval.  Each value is the list of extra flags
# to pass to ALCE-main/eval.py (in addition to --f <file>).
ALCE_FLAGS = {
    "none": None,                                           # skip
    "qa":   ["--qa", "--no_rouge"],                         # str_em + QA-EM/F1
    "full": ["--qa", "--mauve"],                            # + rougeLsum + MAUVE
    "all":  ["--qa", "--mauve", "--citations"],             # + cite-prec/rec (T5-XXL)
}


def run_alce_eval(python_bin: str, alce_dir: str, output_file: str,
                  extra_flags: list, devices: str, log_f) -> dict:
    """Run ALCE eval.py on one output file. Returns parsed .score dict."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = devices
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    cmd = [python_bin, "eval.py", "--f", output_file] + extra_flags
    log_f.write(f"\n$ CUDA_VISIBLE_DEVICES={devices} {' '.join(cmd)}\n")
    log_f.flush()
    proc = subprocess.Popen(cmd, cwd=alce_dir, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line)
        log_f.write(line)
        log_f.flush()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ALCE eval failed (exit {proc.returncode}) on {output_file}")
    score_path = output_file + ".score"
    with open(score_path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="Run ASQA long-form comparison (Original vs LangGraph)")
    ap.add_argument("--num_samples", type=int, default=-1,
                    help="-1 = full 948 samples")
    ap.add_argument("--ndocs", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=300)
    ap.add_argument("--beam_width", type=int, default=2)
    ap.add_argument("--max_depth", type=int, default=7)
    ap.add_argument("--mode", default="always_retrieve",
                    choices=["adaptive_retrieval", "always_retrieve", "no_retrieval"])
    ap.add_argument("--threshold", type=float, default=0.2)

    # ALCE eval controls (only affect post-generation scoring, not the pipelines)
    ap.add_argument("--alce_eval", default="qa", choices=list(ALCE_FLAGS.keys()),
                    help="qa=str_em+QA; full=+rougeLsum+MAUVE (1 GPU); "
                         "all=+citation prec/rec (T5-XXL NLI, needs 2+ GPUs); "
                         "none=skip")
    ap.add_argument("--gen_device", default="0",
                    help="CUDA_VISIBLE_DEVICES for the vLLM generation step (default '0')")
    ap.add_argument("--eval_device", default="0",
                    help="CUDA_VISIBLE_DEVICES for ALCE eval in qa/full mode (default '0')")
    ap.add_argument("--citations_devices", default="0,1",
                    help="CUDA_VISIBLE_DEVICES for --citations (T5-XXL); needs 2+ GPUs (default '0,1')")
    args = ap.parse_args()

    # ---- Paths ----------------------------------------------------------
    base_dir = "/data1/ragworkspace/self-rag"
    work_dir = os.path.join(base_dir, "retrieval_lm")
    output_dir = os.path.join(base_dir, "eval_results")
    log_path = os.path.join(output_dir, "run_asqa_compare.log")
    python_bin = os.path.join(base_dir, ".venv/bin/python")
    input_file = os.path.join(base_dir, "ALCE-main/data/asqa_eval_gtr_top100.json")
    alce_dir = os.path.join(base_dir, "ALCE-main")

    os.makedirs(output_dir, exist_ok=True)

    # ---- Environment for generation -------------------------------------
    env = os.environ.copy()
    env.update({
        "HF_ENDPOINT": "https://hf-mirror.com",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "CUDA_VISIBLE_DEVICES": args.gen_device,
        "TOKENIZERS_PARALLELISM": "false",
    })

    cmd = [
        python_bin, "run_compare_eval_longform.py",
        "--model", "selfrag/selfrag_llama2_7b",
        "--download_dir", os.path.join(base_dir, "model_cache"),
        "--output_dir", output_dir,
        "--input_file", input_file,
        "--task", "asqa",
        "--ndocs", str(args.ndocs),
        "--max_new_tokens", str(args.max_new_tokens),
        "--beam_width", str(args.beam_width),
        "--max_depth", str(args.max_depth),
        "--mode", args.mode,
        "--threshold", str(args.threshold),
        "--w_rel", "1.0",
        "--w_sup", "1.0",
        "--w_use", "0.5",
        "--num_samples", str(args.num_samples),
        "--dtype", "half",
    ]

    with open(log_path, "w") as log_f:
        header = f"===== starting {datetime.now()} =====\n"
        header += f"NUM_SAMPLES={args.num_samples}\n"
        header += f"ALCE_EVAL={args.alce_eval}\n"
        log_f.write(header)
        log_f.flush()
        print(header, end="")

        # ---- 1. Generation + agreement report ---------------------------
        proc = subprocess.Popen(
            cmd, cwd=work_dir, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            log_f.write(line)
            log_f.flush()
        proc.wait()
        if proc.returncode != 0:
            footer = f"===== generation failed (exit {proc.returncode}) at {datetime.now()} =====\n"
            log_f.write(footer)
            print(footer, end="")
            sys.exit(proc.returncode)

        # ---- 2. ALCE evaluation (optional) ------------------------------
        extra_flags = ALCE_FLAGS[args.alce_eval]
        if extra_flags is None:
            print("\n[skip] --alce_eval=none, not running ALCE eval.py")
            log_f.write("\n[skip] --alce_eval=none\n")
        else:
            orig_path = os.path.join(output_dir, "orig_asqa_output.json")
            lg_path   = os.path.join(output_dir, "lg_asqa_output.json")
            # citations uses T5-XXL (11B params, ~22 GB in bf16) -> 2 GPUs
            devices = args.citations_devices if args.alce_eval == "all" else args.eval_device
            print(f"\n===== ALCE eval (mode={args.alce_eval}, devices={devices}) =====")
            log_f.write(f"\n===== ALCE eval (mode={args.alce_eval}, devices={devices}) =====\n")

            try:
                orig_score = run_alce_eval(python_bin, alce_dir, orig_path, extra_flags, devices, log_f)
                lg_score   = run_alce_eval(python_bin, alce_dir, lg_path,   extra_flags, devices, log_f)
            except Exception as e:
                err = f"\nALCE eval failed: {e}\n"
                log_f.write(err)
                print(err, end="")
                sys.exit(1)

            # Pretty summary
            keys = [k for k in [
                "str_em", "str_hit", "rougeLsum",
                "QA-EM", "QA-F1", "QA-Hit",
                "mauve", "citation_prec", "citation_rec",
            ] if k in orig_score]
            summary = "\n===== Final ALCE scores =====\n"
            summary += f"{'metric':<15} {'original':>10} {'langgraph':>10}\n"
            for k in keys:
                summary += f"{k:<15} {orig_score[k]:>10.4f} {lg_score[k]:>10.4f}\n"
            log_f.write(summary)
            print(summary, end="")

        footer = f"===== done {datetime.now()} =====\n"
        log_f.write(footer)
        print(footer, end="")

    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
