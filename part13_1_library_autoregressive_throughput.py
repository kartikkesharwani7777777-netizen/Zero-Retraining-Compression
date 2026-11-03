"""Benchmark WikiText-2 autoregressive throughput with vLLM."""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

import argparse
import threading
import time

import pynvml
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed


MODEL_PATH = str(PROJECT_ROOT / "models" / 'Prune_non_latin' / 'SW3BW70X')
DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_SPLIT = "test"
MAX_LENGTH = 512
MAX_NEW_TOKENS = 128
MAX_WINDOWS = 15
SEED = 42


class NvmlMemoryPeak:
    """Sample total device memory use through NVML while generation runs."""

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
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument(
        "--stride", type=int, default=MAX_LENGTH,
        help="Token step between prompt windows (defaults to non-overlapping windows).",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()
    if args.max_length < 1 or args.stride < 1:
        parser.error("max-length and stride must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("gpu-memory-utilization must be in (0, 1]")
    if args.tensor_parallel_size < 1:
        parser.error("tensor-parallel-size must be positive")

    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        parser.exit(
            2,
            "vLLM is required for this benchmark. Install it with:\n"
            "  python -m pip install vllm\n"
            f"Import error: {exc}\n",
        )

    set_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled NVIDIA GPU is required for this benchmark.")

    device_index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_index)
    pynvml.nvmlInit()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
        full_text = "\n".join(row["text"] for row in dataset)
        token_ids = tokenizer(
            full_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]
        if token_ids.numel() == 0:
            raise RuntimeError("The configured dataset split produced no tokens.")

        windows = [
            token_ids[start : start + args.max_length]
            for start in range(0, token_ids.numel(), args.stride)
        ][:MAX_WINDOWS]
        # Pass token IDs directly so each prompt is exactly the same window as Part 1.1.
        prompts = [{"prompt_token_ids": window.tolist()} for window in windows]
        if not prompts:
            raise RuntimeError("No prompt windows were created.")

        try:
            llm = LLM(
                model=args.model_path,
                trust_remote_code=True,
                dtype="auto",
                max_model_len=args.max_length + MAX_NEW_TOKENS,
                gpu_memory_utilization=args.gpu_memory_utilization,
                tensor_parallel_size=args.tensor_parallel_size,
                seed=SEED,
            )
        except ImportError as exc:
            raise RuntimeError(
                "vLLM could not load the model. Check its model architecture and vLLM compatibility."
            ) from exc

        sampling = SamplingParams(
            temperature=0,
            min_tokens=MAX_NEW_TOKENS,
            max_tokens=MAX_NEW_TOKENS,
        )
        # Pay engine startup and compilation costs before timing and NVML sampling.
        llm.generate([prompts[0]], SamplingParams(temperature=0, max_tokens=2), use_tqdm=False)
        torch.cuda.synchronize(device_index)

        meter = NvmlMemoryPeak(device_index)
        meter.start()
        torch.cuda.synchronize(device_index)
        start_time = time.perf_counter()
        try:
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            torch.cuda.synchronize(device_index)
        finally:
            elapsed_seconds = time.perf_counter() - start_time
            meter.stop()

        generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        print(f"Model: {args.model_path}")
        print("Inference library: vLLM")
        print(f"GPU: {device_name}")
        print(f"Test split tokens: {token_ids.numel():,}")
        print(
            f"Prompt windows: {len(windows):,} (size <= {args.max_length}, "
            f"stride {args.stride}, capped at {MAX_WINDOWS})"
        )
        print(f"Generation length per window: {MAX_NEW_TOKENS} tokens (KV cache enabled)")
        print(f"Generated tokens: {generated_tokens:,}")
        print(f"End-to-end inference time: {elapsed_seconds:.2f} seconds")
        print(f"End-to-end inference throughput: {generated_tokens / elapsed_seconds:.2f} tokens/second")
        print(f"NVML peak GPU memory: {meter.peak_bytes / (1024**3):.2f} GiB")
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
