"""Measure WikiText-2 quality with LM Evaluation Harness and vLLM."""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

import argparse
import threading
import time

import pynvml
import torch


MODEL_PATH = ( PROJECT_ROOT / "models" / "Llama 3.1 8B_models" / "Llama31_Wanda" / "H12W50")
class NvmlMemoryPeak:
    """Sample total device memory use through NVML during evaluation."""

    def __init__(self, device_index, interval_seconds=0.01):
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self.running = False
        self.thread = None

    def _sample(self):
        while self.running:
            self.peak_bytes = max(
                self.peak_bytes,
                pynvml.nvmlDeviceGetMemoryInfo(self.handle).used,
            )
            time.sleep(self.interval_seconds)

    def start(self):
        self.peak_bytes = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used
        self.running = True
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
        self.peak_bytes = max(
            self.peak_bytes,
            pynvml.nvmlDeviceGetMemoryInfo(self.handle).used,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()
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
            "LM Evaluation Harness with vLLM support is required. Install it with:\n"
            '  python -m pip install "lm_eval[vllm]"\n'
            f"Import error: {exc}\n",
        )

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled NVIDIA GPU is required for this benchmark.")
    device_index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_index)

    pynvml.nvmlInit()
    meter = NvmlMemoryPeak(device_index)
    try:
        torch.cuda.synchronize(device_index)
        meter.start()
        try:
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
                tasks=["wikitext"],
                num_fewshot=0,
                batch_size=args.batch_size,
                log_samples=False,
            )
            torch.cuda.synchronize(device_index)
        finally:
            meter.stop()

        if results is None or "wikitext" not in results.get("results", {}):
            raise RuntimeError("The evaluation harness returned no WikiText results.")

        metrics = results["results"]["wikitext"]
        print(f"Model: {args.model_path}")
        print("Dataset: WikiText-2 raw test split (LM Evaluation Harness wikitext task)")
        print(f"Device: {device_name}")
        print("Scoring: rolling log-likelihood (LM Evaluation Harness with vLLM)")
        for metric in ("word_perplexity,none", "byte_perplexity,none", "bits_per_byte,none"):
            if metric in metrics:
                print(f"{metric.split(',')[0].replace('_', ' ').title()}: {metrics[metric]:.4f}")
        print(f"NVML peak GPU memory: {meter.peak_bytes / (1024**3):.2f} GiB")
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
