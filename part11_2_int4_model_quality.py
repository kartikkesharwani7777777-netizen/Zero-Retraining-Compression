"""Measure WikiText-2 test perplexity with TorchAO INT4 weight-only quantization."""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent


import os
from pathlib import Path


import argparse
import math
import threading
import time

import pynvml
import torch
from torchao.quantization import Int4WeightOnlyConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig


MODEL_PATH = str(PROJECT_ROOT / "models" / "Prune_non_latin" / "PureX")
DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_SPLIT = "test"
MAX_LENGTH = 2048
STRIDE = 2048

# Use TorchAO's Tensor Core tiled INT4 packing with TinyGEMM qparams/kernel path.
INT4_CONFIG = Int4WeightOnlyConfig(
    int4_packing_format="tile_packed_to_4d",
    int4_choose_qparams_algorithm="tinygemm",
)


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
            self.peak_bytes = max(self.peak_bytes, pynvml.nvmlDeviceGetMemoryInfo(self.handle).used)
            time.sleep(self.interval_seconds)

    def start(self):
        self.peak_bytes = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used
        self.running = True
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
        self.peak_bytes = max(self.peak_bytes, pynvml.nvmlDeviceGetMemoryInfo(self.handle).used)


def measure_perplexity(model, token_ids, max_length, stride, device):
    """Score each token once, using preceding tokens as context when available."""
    total_nll = 0.0
    total_scored_tokens = 0
    previous_end = 0
    for start in range(0, token_ids.numel(), stride):
        end = min(start + max_length, token_ids.numel())
        input_ids = token_ids[start:end].unsqueeze(0).to(device)
        target_start = max(previous_end - start, 0)
        labels = input_ids.clone()
        labels[:, :target_start] = -100
        scored = int((labels[:, 1:] != -100).sum().item())
        if scored:
            with torch.inference_mode():
                output = model(input_ids=input_ids, labels=labels, use_cache=False)
            total_nll += output.loss.float().item() * scored
            total_scored_tokens += scored
        previous_end = end
        if end == token_ids.numel():
            break
    if total_scored_tokens == 0:
        raise RuntimeError("The dataset did not contain enough tokens to score perplexity.")
    return math.exp(total_nll / total_scored_tokens), total_scored_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--stride", type=int, default=STRIDE)
    args = parser.parse_args()
    if args.max_length < 2 or args.stride < 1 or args.stride >= args.max_length:
        parser.error("max-length must be at least 2 and stride must be between 1 and max-length - 1")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled NVIDIA GPU is required for INT4 GPU evaluation.")

    device_index = torch.cuda.current_device()
    device = torch.device(f"cuda:{device_index}")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            quantization_config=TorchAoConfig(quant_type=INT4_CONFIG),
            device_map={"": str(device)},
        )
    except ImportError as error:
        raise RuntimeError(
            "INT4 loading requires torchao and Accelerate. Install them with: "
            "python -m pip install torchao accelerate"
        ) from error
    model.eval()
    # Compile after loading the INT4 weights; run one forward pass before scoring.
    model = torch.compile(model)

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    full_text = "\n".join(row["text"] for row in dataset)
    token_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if token_ids.numel() < 2:
        raise RuntimeError("The configured WikiText-2 test split produced fewer than two tokens.")

    pynvml.nvmlInit()
    meter = NvmlMemoryPeak(device_index)
    warmup_length = min(args.max_length, token_ids.numel())
    warmup_ids = token_ids[:warmup_length].unsqueeze(0).to(
        model.get_input_embeddings().weight.device
    )
    with torch.inference_mode():
        model(input_ids=warmup_ids, use_cache=False)
    torch.cuda.synchronize(device_index)
    meter.start()
    try:
        perplexity, scored_tokens = measure_perplexity(
            model, token_ids, args.max_length, args.stride, model.get_input_embeddings().weight.device
        )
        torch.cuda.synchronize(device_index)
    finally:
        meter.stop()
        pynvml.nvmlShutdown()
    print(f"Model: {args.model_path}")
    print("Weight quantization: INT4 weight-only (TorchAO Tensor Core tiled / TinyGEMM)")
    print("Model execution: torch.compile")
    print(f"Dataset: {DATASET_NAME}/{DATASET_CONFIG} ({DATASET_SPLIT})")
    print(f"Device: {device}")
    print(f"Scored tokens: {scored_tokens:,}")
    print(f"WikiText-2 test perplexity: {perplexity:.4f}")
    print(f"NVML peak GPU memory: {meter.peak_bytes / (1024**3):.2f} GiB")


if __name__ == "__main__":
    main()
