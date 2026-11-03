"""Part 8.1: faster batched stratified pruning of Qwen3.5 decoder groups."""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent


import os
from pathlib import Path


from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# Edit these settings.
MODEL_PATH = str(PROJECT_ROOT / "models" / 'hub' / 'Qwen3.5-9B')
OUTPUT_PATH = str(PROJECT_ROOT / "models" / 'Prune_stratified' / 'SW2B')
PRUNING_PERCENTAGE = 40
GROUP_SIZE = 4  # Three Gated DeltaNet blocks and one full-attention block.
CALIBRATION_EXAMPLES = 32
MAX_LENGTH = 512
BATCH_SIZE = 4


def find_decoder(model):
    """Return the module that owns the decoder layer list."""
    for module in model.modules():
        layers = getattr(module, "layers", None)
        if isinstance(layers, nn.ModuleList) and layers and "DecoderLayer" in type(layers[0]).__name__:
            return module
    raise TypeError("Could not find the text decoder in this causal language model.")


def classify_layer(block, configured_type=None):
    """Identify a hybrid decoder block using config metadata or its modules."""
    kind = str(configured_type or "").lower()
    if "linear" in kind or "deltanet" in kind:
        return "deltanet"
    if "full" in kind:
        return "full_attention"
    if any(getattr(block, name, None) is not None for name in ("linear_attn", "linear_attention", "gated_deltanet")):
        return "deltanet"
    if any(getattr(block, name, None) is not None for name in ("self_attn", "attention", "attn")):
        return "full_attention"
    raise TypeError(f"Could not classify decoder block {type(block).__name__}.")


def find_candidate_groups(kinds, group_size):
    """Find complete non-boundary groups containing exactly three DeltaNet blocks."""
    groups = []
    for start in range(0, len(kinds) - group_size + 1, group_size):
        indices = tuple(range(start, start + group_size))
        selected = [kinds[index] for index in indices]
        if selected.count("deltanet") == 3 and selected.count("full_attention") == 1:
            groups.append(indices)
    return groups[1:-1]  # Preserve the first and final complete groups.


def update_layer_metadata(model, decoder, retained):
    """Keep decoder counts and hybrid layer-type metadata aligned after pruning."""
    configs = {id(config): config for config in (
        model.config, getattr(model.config, "text_config", None), decoder.config
    ) if config is not None}.values()
    for config in configs:
        if hasattr(config, "num_hidden_layers"):
            config.num_hidden_layers = len(retained)
        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None:
            config.layer_types = [layer_types[index] for index in retained if index < len(layer_types)]


def calibrate_influence(model, tokenizer, layers, dataset):
    """Measure mean tokenwise angular change, batching texts and ignoring padding."""
    device = model.get_input_embeddings().weight.device
    totals = torch.zeros(len(layers), dtype=torch.float64)
    counts = torch.zeros(len(layers), dtype=torch.float64)
    active_mask = None
    handles = []

    def make_hook(index):
        def score(_module, inputs, output):
            
            block_input = inputs[0].detach().float()
            block_output = output[0] if isinstance(output, (tuple, list)) else output
            block_output = block_output.detach().float()
            distance = 1 - F.cosine_similarity(block_input, block_output, dim=-1, eps=1e-8)
            mask = active_mask.to(distance.device)
            totals[index] += (distance * mask).sum().double().cpu()
            counts[index] += mask.sum().cpu()
        return score

    for index, block in enumerate(layers):
        handles.append(block.register_forward_hook(make_hook(index)))
    try:
        texts = [row["text"] for row in dataset if row["text"].strip()]
        with torch.inference_mode():
            for start in range(0, len(texts), BATCH_SIZE):
                batch = tokenizer(
                    texts[start:start + BATCH_SIZE], return_tensors="pt", padding=True,
                    truncation=True, max_length=MAX_LENGTH,
                )
                active_mask = batch["attention_mask"].to(device=device, dtype=torch.float32)
                model(**{key: value.to(device) for key, value in batch.items()}, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    if torch.any(counts == 0):
        raise RuntimeError("Calibration did not capture scores for every decoder block.")
    return (totals / counts).tolist()


def main():
    if GROUP_SIZE != 4:
        raise ValueError("GROUP_SIZE must be 4 for the 3:1 pruning ratio.")
    if not 0 <= PRUNING_PERCENTAGE < 100:
        raise ValueError("PRUNING_PERCENTAGE must be in [0, 100).")
    if min(CALIBRATION_EXAMPLES, MAX_LENGTH, BATCH_SIZE) < 1:
        raise ValueError("Calibration examples, max length, and batch size must be positive.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype="auto", low_cpu_mem_usage=True, device_map="auto"
    ).eval()
    decoder = find_decoder(model)
    layers = decoder.layers
    configured_types = getattr(getattr(model.config, "text_config", model.config), "layer_types", None)
    kinds = [
        classify_layer(block, configured_types[index] if configured_types and index < len(configured_types) else None)
        for index, block in enumerate(layers)
    ]
    groups = find_candidate_groups(kinds, GROUP_SIZE)
    if not groups:
        raise ValueError("No eligible interior four-block groups with a 3:1 DeltaNet/full-attention mix.")

    remove_count = int(len(groups) * PRUNING_PERCENTAGE / 100)
    if remove_count < 1 or remove_count >= len(groups):
        raise ValueError("Percentage must remove at least one group and leave at least one candidate group.")

    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    data = data.filter(lambda row: bool(row["text"].strip()))
    data = data.select(range(min(CALIBRATION_EXAMPLES, len(data))))
    scores = calibrate_influence(model, tokenizer, layers, data)
    group_scores = [sum(scores[index] for index in group) / GROUP_SIZE for group in groups]
    removed_groups = sorted(sorted(range(len(groups)), key=group_scores.__getitem__)[:remove_count])
    removed = {index for group_id in removed_groups for index in groups[group_id]}
    retained = [index for index in range(len(layers)) if index not in removed]

    decoder.layers = nn.ModuleList(layers[index] for index in retained)
    for index, block in enumerate(decoder.layers):
        for module in block.modules():
            if hasattr(module, "layer_idx"):
                module.layer_idx = index
    update_layer_metadata(model, decoder, retained)

    output = Path(OUTPUT_PATH)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    print(f"Saved stratified-pruned model to: {output.resolve()}")
    print(f"Removed {remove_count}/{len(groups)} eligible groups ({len(removed)} decoder blocks).")
    print(f"Removed original block indices: {', '.join(map(str, sorted(removed)))}")
    print(f"Batched calibration: {len(data)} WikiText examples, batch size {BATCH_SIZE}.")


if __name__ == "__main__":
    main()
