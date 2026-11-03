"""Measure model parameter count and peak PyTorch CUDA allocation on WikiText-2."""

import argparse
import math
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = str(PROJECT_ROOT / "models" /"Llama 3.1 8B_models"/ "Llama31_Wanda" / "H12W50")
DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_SPLIT = "test"
MAX_LENGTH = 512
STRIDE = 256


def measure_quality(model, token_ids, token_byte_lengths, max_length, stride, device):
    """Return perplexity and bits per UTF-8 byte for the scored tokens."""
    total_nll = 0.0
    total_scored_tokens = 0
    total_scored_bytes = 0
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
            first_scored_token = start + max(target_start + 1, 1)
            total_scored_bytes += sum(token_byte_lengths[first_scored_token:end])
        previous_end = end
        if end == token_ids.numel():
            break

    if total_scored_tokens == 0:
        raise RuntimeError("The dataset did not contain enough tokens to score perplexity.")
    if total_scored_bytes == 0:
        raise RuntimeError("No UTF-8 bytes were aligned with the scored tokens.")
    perplexity = math.exp(total_nll / total_scored_tokens)
    bits_per_byte = total_nll / math.log(2) / total_scored_bytes
    return perplexity, bits_per_byte, total_scored_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--stride", type=int, default=STRIDE)
    args = parser.parse_args()
    if args.max_length < 2 or args.stride < 1 or args.stride > args.max_length:
        parser.error("max-length must be at least 2 and stride must be between 1 and max-length")

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled NVIDIA GPU is required for CUDA allocation measurement.")
    device_index = torch.cuda.current_device()
    device = torch.device(f"cuda:{device_index}")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map={"": str(device)},
        attn_implementation="sdpa",
    )
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    full_text = "\n".join(row["text"] for row in dataset)
    if not tokenizer.is_fast:
        raise RuntimeError("Bits-per-byte measurement requires a fast tokenizer with offset mappings.")
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    token_ids = encoded["input_ids"][0]
    offsets = encoded["offset_mapping"][0].tolist()
    token_byte_lengths = [len(full_text[start:end].encode("utf-8")) for start, end in offsets]
    if token_ids.numel() < 2:
        raise RuntimeError("The configured WikiText-2 test split produced fewer than two tokens.")

    # Establish a fresh peak for this benchmark; the current model allocation is
    # retained as the baseline, while cached but unused memory is excluded.
    torch.cuda.synchronize(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)
    perplexity, bits_per_byte, scored_tokens = measure_quality(
        model,
        token_ids,
        token_byte_lengths,
        args.max_length,
        args.stride,
        model.get_input_embeddings().weight.device,
    )
    torch.cuda.synchronize(device_index)
    peak_allocated_gib = torch.cuda.max_memory_allocated(device_index) / (1024**3)

    print(f"Model: {args.model_path}")
    print(f"GPU: {torch.cuda.get_device_name(device_index)}")
    print(f"Parameter count: {parameter_count:,}")
    print(f"Peak PyTorch CUDA allocation: {peak_allocated_gib:.2f} GiB")
    print(f"Scored tokens: {scored_tokens:,}")
    print(f"WikiText-2 test perplexity: {perplexity:.4f}")
    print(f"Bits per byte (BPB): {bits_per_byte:.6f}")


if __name__ == "__main__":
    main()
