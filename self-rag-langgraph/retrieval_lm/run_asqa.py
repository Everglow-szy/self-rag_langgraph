"""Entry point: run ASQA long-form comparison experiment.

Usage:
    python run_asqa.py
    python run_asqa.py --num_samples 50
"""
import os
import sys
import subprocess
import argparse
from datetime import datetime


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
    args = ap.parse_args()

    # Paths
    base_dir = "/data1/ragworkspace/self-rag"
    work_dir = os.path.join(base_dir, "retrieval_lm")
    output_dir = os.path.join(base_dir, "eval_results")
    log_path = os.path.join(output_dir, "run_asqa_compare.log")
    python_bin = os.path.join(base_dir, ".venv/bin/python")
    input_file = os.path.join(base_dir, "ALCE-main/data/asqa_eval_gtr_top100.json")

    os.makedirs(output_dir, exist_ok=True)

    # Environment
    env = os.environ.copy()
    env.update({
        "HF_ENDPOINT": "https://hf-mirror.com",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "CUDA_VISIBLE_DEVICES": "0",
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
        log_f.write(header)
        log_f.flush()
        print(header, end="")

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

        footer = f"===== done {datetime.now()} =====\n"
        log_f.write(footer)
        print(footer, end="")

    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
