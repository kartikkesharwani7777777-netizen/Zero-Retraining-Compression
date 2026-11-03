"""Part 5: remove Gated DeltaNet blocks while retaining every other block."""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent


import os
from pathlib import Path


from pathlib import Path

from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# Edit these settings.
MODEL_PATH = str(PROJECT_ROOT / "models" / "Prune_hidden" / "A50H6")
PRUNING_PERCENTAGE = 25  # Percentage of Gated DeltaNet blocks to remove (0-100).
OUTPUT_PATH = str(PROJECT_ROOT / "models" / 'deltanet' / 'deltanet_pruned_model')
def is_gated_deltanet(block):
    """Recognize Gated DeltaNet blocks without matching full-attention blocks."""
    block_type = type(block).__name__.lower()
    if "deltanet" in block_type:
        return True

    for name in ("linear_attn", "self_attn", "attn"):
        attention = getattr(block, name, None)
        if attention is not None and "deltanet" in type(attention).__name__.lower():
            return True

    # Qwen3.5's linear attention module is implemented as Gated DeltaNet and
    # exposed as `linear_attn` on its decoder block.
    return getattr(block, "linear_attn", None) is not None


def main():
    if not 0 <= PRUNING_PERCENTAGE < 100:
        raise ValueError("PRUNING_PERCENTAGE must be at least 0 and less than 100.")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype="auto", low_cpu_mem_usage=True
    )

    decoder = None
    for module in model.modules():
        layers = getattr(module, "layers", None)
        if isinstance(layers, nn.ModuleList) and layers:
            matching = [is_gated_deltanet(block) for block in layers]
            if any(matching):
                decoder = module
                break
    if decoder is None:
        raise TypeError("Could not find decoder layers containing Gated DeltaNet blocks.")

    layers = decoder.layers
    delta_indices = [index for index, block in enumerate(layers) if is_gated_deltanet(block)]
    remove_count = int(len(delta_indices) * PRUNING_PERCENTAGE / 100)
    if remove_count == 0:
        raise ValueError("PRUNING_PERCENTAGE rounds down to zero removable Gated DeltaNet blocks.")
    if remove_count >= len(delta_indices):
        raise ValueError("Pruning must leave at least one Gated DeltaNet block.")

    # Spread removals across the network so the retained DeltaNet blocks are
    # distributed among the full-attention blocks. Other blocks are untouched.
    remove_positions = {
        ((position + 1) * len(delta_indices)) // (remove_count + 1)
        for position in range(remove_count)
    }
    removed_indices = {delta_indices[position] for position in remove_positions}
    retained_indices = [index for index in range(len(layers)) if index not in removed_indices]
    decoder.layers = nn.ModuleList(
        layers[index] for index in retained_indices
    )

    # The remaining layer count must be recorded so Transformers can reload
    # the shortened decoder. No head counts or other layer dimensions change.
    configs = {
        id(config): config
        for config in (model.config, getattr(model.config, "text_config", None), decoder.config)
        if config is not None
    }.values()
    for config in configs:
        if config is not None and hasattr(config, "num_hidden_layers"):
            config.num_hidden_layers = len(decoder.layers)
        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None:
            config.layer_types = [
                layer_types[index]
                for index in retained_indices
                if index < len(layer_types)
            ]

    output = Path(OUTPUT_PATH)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    AutoTokenizer.from_pretrained(MODEL_PATH).save_pretrained(output)

    actual_percentage = remove_count / len(delta_indices) * 100
    print(f"Saved Gated DeltaNet-pruned model to: {output.resolve()}")
    print(
        f"Removed {remove_count} of {len(delta_indices)} Gated DeltaNet blocks "
        f"({actual_percentage:.1f}%, requested {PRUNING_PERCENTAGE}%). "
        f"All other decoder blocks and attention head dimensions were left unchanged."
    )


if __name__ == "__main__":
    main()
