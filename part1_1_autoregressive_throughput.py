"""Benchmark end-to-end inference throughput on WikiText-2 test windows."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


import os
from pathlib import Path


import argparse
import threading
import time

import pynvml
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


# Keep the model and dataset settings aligned with Part 1.
MODEL_PATH = str(PROJECT_ROOT / "models" / 'Prune_non_latin' / 'SW3BW70X')
#MODEL_PATH = (    PROJECT_ROOT / "models" / "Llama 3.1 8B_models" / "Llama31_LayerPruned" / "P60")
#MODEL_PATH = ( PROJECT_ROOT / "models" / "Llama 3.1 8B_models" / "Llama31_Wanda" / "H12W50")


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
        "--stride",
        type=int,
        default=MAX_LENGTH,
        help="Token step between prompt windows (defaults to non-overlapping windows).",
    )
    args = parser.parse_args()
    if args.max_length < 1 or args.stride < 1:
        parser.error("max-length and stride must be positive")

    set_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled NVIDIA GPU is required for this benchmark.")

    device_index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_index)
    pynvml.nvmlInit()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
            device_map={"": f"cuda:{device_index}"},
        )
        model.eval()
        input_device = model.get_input_embeddings().weight.device

        dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
        # Join first, then tokenize once so each window is taken from one continuous
        # token stream spanning every row in the test split.
        full_text = "\n".join(row["text"] for row in dataset)
        token_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        if token_ids.numel() == 0:
            raise RuntimeError("The configured dataset split produced no tokens.")

        windows = [
            token_ids[start : start + args.max_length]
            for start in range(0, token_ids.numel(), args.stride)
        ]
        windows = windows[:MAX_WINDOWS]
        # Warm up generation outside both the timer and NVML measurement.
        warmup = windows[0][: min(windows[0].numel(), args.max_length)].unsqueeze(0).to(input_device)
        with torch.inference_mode():
            model.generate(
                warmup,
                max_new_tokens=min(MAX_NEW_TOKENS, 2),
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize(device_index)

        generated_tokens = 0
        meter = NvmlMemoryPeak(device_index)
        meter.start()
        torch.cuda.synchronize(device_index)
        start_time = time.perf_counter()
        try:
            with torch.inference_mode():
                for window in windows:
                    input_ids = window.unsqueeze(0).to(input_device)
                    output = model.generate(
                        input_ids,
                        min_new_tokens=MAX_NEW_TOKENS,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    generated_tokens += output.shape[-1] - input_ids.shape[-1]
            torch.cuda.synchronize(device_index)
        finally:
            elapsed_seconds = time.perf_counter() - start_time
            meter.stop()

        print(f"Model: {args.model_path}")
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
