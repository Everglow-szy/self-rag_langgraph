"""Entry point: run PopQA short-form comparison experiment.

Usage:
    python run_popqa.py
    python run_popqa.py --num_samples 100 --ndocs 5 --mode adaptive_retrieval
"""
import os
import sys
import subprocess
import argparse
from datetime import datetime


def main():
    ap = argparse.ArgumentParser(description="Run PopQA short-form comparison (Original vs LangGraph)")
    ap.add_argument("--num_samples", type=int, default=-1,
                    help="-1 = full 1,399 long-tail subset")
    ap.add_argument("--ndocs", type=int, default=5)
    ap.add_argument("--mode", default="adaptive_retrieval",
                    choices=["adaptive_retrieval", "always_retrieve", "no_retrieval"])
    ap.add_argument("--tag", default="longtail")
    args = ap.parse_args()

    # Paths
    base_dir = "/data1/ragworkspace/self-rag"
    work_dir = os.path.join(base_dir, "retrieval_lm")
    output_dir = os.path.join(base_dir, "eval_results")
    log_path = os.path.join(output_dir, f"run_{args.tag}.log")
    python_bin = os.path.join(base_dir, ".venv/bin/python")

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
        python_bin, "run_compare_eval.py",
        "--model", "selfrag/selfrag_llama2_7b",
        "--download_dir", os.path.join(base_dir, "model_cache"),
        "--output_dir", output_dir,
        "--num_samples", str(args.num_samples),
        "--ndocs", str(args.ndocs),
        "--max_new_tokens", "100",
        "--mode", args.mode,
        "--w_rel", "1.0",
        "--w_sup", "1.0",
        "--w_use", "0.5",
    ]

    with open(log_path, "w") as log_f:
        header = f"===== starting {datetime.now()} =====\n"
        header += f"NUM_SAMPLES={args.num_samples} NDOCS={args.ndocs} MODE={args.mode}\n"
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
