"""Measure INT4 model size, peak CUDA allocation, perplexity, and BPB on WikiText-2."""

import argparse
import math
from pathlib import Path

import torch
from datasets import load_dataset
from torchao.quantization import Int4WeightOnlyConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = str(PROJECT_ROOT / "models" / "Prune_non_latin" / "PureX")
DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_SPLIT = "test"
MAX_LENGTH = 512
STRIDE = 256

# Match Part 11.2's Tensor Core tiled INT4 packing and TinyGEMM qparams/kernel.
INT4_CONFIG = Int4WeightOnlyConfig(
    int4_packing_format="tile_packed_to_4d",
    int4_choose_qparams_algorithm="tinygemm",
)


def get_static_weight_bytes(model):
    """Sum unique parameter and buffer storage, including quantized storage."""
    seen_storages = set()
    total_bytes = 0

    def add_tensor_storage(tensor):
        nonlocal total_bytes
        if tensor.device.type == "meta":
            return
        # TorchAO quantized tensors may store data and qparams in component tensors.
        flatten = getattr(tensor, "__tensor_flatten__", None)
        if flatten is not None:
            names, _ = flatten()
            if names:
                for name in names:
                    add_tensor_storage(getattr(tensor, name))
                return
        storage = tensor.untyped_storage()
        key = (tensor.device.type, tensor.device.index, storage.data_ptr())
        if key not in seen_storages:
            seen_storages.add(key)
            total_bytes += storage.nbytes()

    for tensor in list(model.parameters()) + list(model.buffers()):
        add_tensor_storage(tensor)
    return total_bytes


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
            first_scored_token = start + max(target_start, 1)
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
        raise RuntimeError("A CUDA-enabled NVIDIA GPU is required for INT4 CUDA allocation measurement.")

    device_index = torch.cuda.current_device()
    device = torch.device(f"cuda:{device_index}")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print("Loading model and applying TorchAO INT4 weight-only quantization...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            quantization_config=TorchAoConfig(quant_type=INT4_CONFIG),
            device_map={"": str(device)},
            attn_implementation="sdpa",
        )
    except ImportError as error:
        raise RuntimeError(
            "INT4 loading requires torchao and Accelerate. Install them with: "
            "python -m pip install torchao accelerate"
        ) from error

    static_weight_bytes = get_static_weight_bytes(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model.eval()
    model = torch.compile(model)

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

    input_device = model.get_input_embeddings().weight.device
    warmup_ids = token_ids[: min(args.max_length, token_ids.numel())].unsqueeze(0).to(input_device)
    # This is measure_quality()'s first window: target_start is zero, so all
    # labels remain enabled and match this warm-up call exactly.
    warmup_labels = warmup_ids.clone()
    with torch.inference_mode():
        model(input_ids=warmup_ids, labels=warmup_labels, use_cache=False)
    torch.cuda.synchronize(device_index)

    # Reset after compile warm-up, immediately before the measured evaluation.
    torch.cuda.reset_peak_memory_stats(device_index)
    perplexity, bits_per_byte, scored_tokens = measure_quality(
        model,
        token_ids,
        token_byte_lengths,
        args.max_length,
        args.stride,
        input_device,
    )
    torch.cuda.synchronize(device_index)
    peak_allocated_gib = torch.cuda.max_memory_allocated(device_index) / (1024**3)

    print(f"Model: {args.model_path}")
    print("Weight quantization: INT4 weight-only (TorchAO Tensor Core tiled / TinyGEMM)")
    print("Model execution: torch.compile")
    print(f"GPU: {torch.cuda.get_device_name(device_index)}")
    print(f"Parameter count: {parameter_count:,}")
    print(f"Static weight storage after loading: {static_weight_bytes / (1024**3):.2f} GiB")
    print(f"Peak PyTorch CUDA allocation: {peak_allocated_gib:.2f} GiB")
    print(f"Scored tokens: {scored_tokens:,}")
    print(f"WikiText-2 test perplexity: {perplexity:.4f}")
    print(f"Bits per byte (BPB): {bits_per_byte:.6f}")


if __name__ == "__main__":
    main()
