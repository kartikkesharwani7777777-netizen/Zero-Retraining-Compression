"""Benchmark WikiText quality and GPU performance with INT8 model weights."""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent


import os
from pathlib import Path


import argparse
import threading
import time

import pynvml
import torch
from torchao.quantization import Int8WeightOnlyConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig, set_seed


# Keep the workload aligned with part1_Quick_test_model.py.
MODEL_PATH = str(PROJECT_ROOT / "models" / "Prune_non_latin" / "SW3BX")
DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_SPLIT = "test"
MAX_EXAMPLES = 100
MAX_LENGTH = 512
SEED = 42


print(MODEL_PATH)

class GpuEnergyMeter:
    """Sample NVIDIA GPU power and integrate it to estimate energy in joules."""

    def __init__(self, device_index, interval_seconds=0.05):
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.interval_seconds = interval_seconds
        self.running = False
        self.energy_joules = 0.0
        self._thread = None

    def _sample(self):
        previous_time = time.perf_counter()
        previous_power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        while self.running:
            time.sleep(self.interval_seconds)
            current_time = time.perf_counter()
            current_power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
            self.energy_joules += (previous_power + current_power) * 0.5 * (
                current_time - previous_time
            )
            previous_time, previous_power = current_time, current_power

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        self._thread.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=MODEL_PATH)
    args = parser.parse_args()
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled NVIDIA GPU is required for INT8 GPU inference.")

    device_index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_index)
    try:
        pynvml.nvmlInit()
        pynvml.nvmlDeviceGetPowerUsage(pynvml.nvmlDeviceGetHandleByIndex(device_index))
    except pynvml.NVMLError as error:
        raise RuntimeError(
            "Could not read NVIDIA GPU power. Install the NVIDIA driver and "
            "nvidia-ml-py package, then try again."
        ) from error

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            quantization_config=TorchAoConfig(quant_type=Int8WeightOnlyConfig()),
            device_map={"": f"cuda:{device_index}"},
        )
    except ImportError as error:
        raise RuntimeError(
            "INT8 loading requires torchao and Accelerate. Install them with: "
            "python -m pip install torchao accelerate"
        ) from error
    model.eval()
    input_device = model.get_input_embeddings().weight.device

    data = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    data = data.filter(lambda row: bool(row["text"].strip()))
    data = data.select(range(min(MAX_EXAMPLES, len(data))))
    if not data:
        raise RuntimeError("The configured WikiText split contains no non-empty examples.")
    examples = [
        tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        for row in data
    ]

    warmup = {key: value.to(input_device) for key, value in examples[0].items()}
    with torch.inference_mode():
        model(**warmup, use_cache=False)
    torch.cuda.synchronize(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)

    token_count = 0
    negative_log_likelihood = 0.0
    elapsed_seconds = 0.0
    meter = GpuEnergyMeter(device_index)
    meter.start()
    try:
        with torch.inference_mode():
            for encoded in examples:
                batch = {key: value.to(input_device) for key, value in encoded.items()}
                labels = batch["input_ids"].clone()
                if "attention_mask" in batch:
                    labels[batch["attention_mask"] == 0] = -100
                    token_count += int(batch["attention_mask"].sum().item()) - 1
                else:
                    token_count += labels.numel() - 1

                torch.cuda.synchronize(device_index)
                start = time.perf_counter()
                output = model(**batch, labels=labels, use_cache=False)
                torch.cuda.synchronize(device_index)
                elapsed_seconds += time.perf_counter() - start
                scored_tokens = int((labels[:, 1:] != -100).sum().item())
                negative_log_likelihood += output.loss.float().item() * scored_tokens
    finally:
        torch.cuda.synchronize(device_index)
        meter.stop()
        pynvml.nvmlShutdown()

    if token_count <= 0 or elapsed_seconds <= 0:
        raise RuntimeError("No tokens were available for the benchmark.")
    perplexity = torch.exp(torch.tensor(negative_log_likelihood / token_count)).item()
    peak_memory_gb = torch.cuda.max_memory_reserved(device_index) / (1024**3)
    print(f"Model: {args.model_path}")
    print("Weight quantization: INT8 weight-only (TorchAO)")
    print(f"GPU: {device_name}")
    print(f"Model quality (WikiText perplexity): {perplexity:.2f}")
    print(
        f"Inference speed: {token_count / elapsed_seconds:.2f} tokens/second; "
        f"{elapsed_seconds / len(examples):.3f} seconds/example"
    )
    print(f"GPU memory peak: {peak_memory_gb:.2f} GB reserved")
    print(
        f"GPU energy during benchmark: {meter.energy_joules:.2f} joules "
        f"({meter.energy_joules / 3600:.5f} Wh)"
    )


if __name__ == "__main__":
    main()
