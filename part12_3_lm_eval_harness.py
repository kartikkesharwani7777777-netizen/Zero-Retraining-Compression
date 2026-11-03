"""Run Part 12 multiple-choice benchmarks with LM Evaluation Harness and vLLM."""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

import argparse


MODEL_PATH = str(PROJECT_ROOT / "models" / 'Prune_non_latin' / 'SW2BX')
TASKS = ["arc_challenge", "winogrande", "hellaswag"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optional limit per task (useful for a quick run).")
    parser.add_argument("--batch-size", default="auto",
                        help="Harness batch size, such as 1, 4, or auto (default: auto).")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="Fraction of GPU memory vLLM may use (default: 0.9).")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of GPUs for tensor parallel inference (default: 1).")
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if args.max_length < 2:
        parser.error("--max-length must be at least 2")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be positive")

    try:
        import lm_eval
    except ImportError as exc:
        parser.exit(
            2,
            "LM Evaluation Harness is required. Install its Hugging Face backend with:\n"
            '  python -m pip install "lm_eval[vllm]"\n'
            f"Import error: {exc}\n",
        )

    print(
        f"Model: {args.model_path}\n"
        f"Tasks: {', '.join(args.tasks)}\n"
        "Harness: EleutherAI LM Evaluation Harness with vLLM; zero-shot"
    )
    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args={
            "pretrained": args.model_path,
            "trust_remote_code": True,
            "dtype": "auto",
            "max_model_len": args.max_length,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tensor_parallel_size": args.tensor_parallel_size,
        },
        tasks=args.tasks,
        num_fewshot=0,
        batch_size=args.batch_size,
        limit=args.max_samples,
        log_samples=False,
    )
    if results is None:
        raise RuntimeError("The harness returned no results.")
    print("\nResults:")
    for task, metrics in results["results"].items():
        print(f"{task}:")
        for metric, value in metrics.items():
            if not metric.endswith(",none"):
                continue
            print(f"  {metric}: {value}")


if __name__ == "__main__":
    main()
