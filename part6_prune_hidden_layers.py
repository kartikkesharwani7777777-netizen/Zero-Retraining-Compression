"""Part 6: remove low-importance decoder blocks using weight magnitude."""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent


import os
from pathlib import Path


from pathlib import Path

from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# Edit these settings.
MODEL_PATH = str(PROJECT_ROOT / "models" / "Prune_attentionhead" / "A50")
PRUNING_PERCENTAGE = 20  # Percentage of decoder blocks to remove (0-100).
OUTPUT_PATH = str(PROJECT_ROOT / "models" / 'Prune_hidden' / 'A50H6')
def find_decoder(model):
    """Find a decoder whose layers are decoder blocks, not vision or other layers."""
    for module in model.modules():
        layers = getattr(module, "layers", None)
        if (
            isinstance(layers, nn.ModuleList)
            and layers
            and "DecoderLayer" in type(layers[0]).__name__
        ):
            return module
    raise TypeError("Could not find the text decoder in this causal language model.")


def block_importance(block):
    """Score a block by mean squared parameter magnitude (RMS squared)."""
    squared_sum = 0.0
    parameter_count = 0
    for parameter in block.parameters():
        values = parameter.detach().float()
        squared_sum += values.square().sum().item()
        parameter_count += values.numel()
    if parameter_count == 0:
        raise ValueError("A decoder block has no parameters to score.")
    return squared_sum / parameter_count


def layer_type(config, index):
    """Return the configured hybrid layer type, when the model exposes it."""
    types = getattr(config, "layer_types", None)
    if types is not None and index < len(types):
        return types[index]
    return None


def main():
    if not 0 <= PRUNING_PERCENTAGE < 100:
        raise ValueError("PRUNING_PERCENTAGE must be at least 0 and less than 100.")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype="auto", low_cpu_mem_usage=True
    )
    decoder = find_decoder(model)
    layers = decoder.layers
    layer_count = len(layers)
    remove_count = int(layer_count * PRUNING_PERCENTAGE / 100)
    if remove_count == 0:
        raise ValueError("PRUNING_PERCENTAGE rounds down to zero removable decoder blocks.")
    if remove_count >= layer_count:
        raise ValueError("Pruning must leave at least one decoder block.")

    configs = list(
        {
            id(config): config
            for config in (model.config, getattr(model.config, "text_config", None), decoder.config)
            if config is not None
        }.values()
    )
    primary_config = getattr(model.config, "text_config", model.config)
    types = getattr(primary_config, "layer_types", None)
    scores = [block_importance(block) for block in layers]

    # Keep at least one layer of every configured type (for hybrid architectures).
    type_counts = {}
    if types is not None:
        for index in range(layer_count):
            kind = layer_type(primary_config, index)
            type_counts[kind] = type_counts.get(kind, 0) + 1
    removable = [
        index for index in range(layer_count)
        if types is None or type_counts[layer_type(primary_config, index)] > 1
    ]
    if remove_count > len(removable):
        raise ValueError(
            f"Cannot remove {remove_count} blocks while retaining at least one of each "
            "configured layer type. Reduce PRUNING_PERCENTAGE."
        )

    removed_indices = set(sorted(removable, key=lambda i: scores[i])[:remove_count])
    retained_indices = [i for i in range(layer_count) if i not in removed_indices]
    decoder.layers = nn.ModuleList(layers[i] for i in retained_indices)

    # Keep all configs consistent with the physically shortened layer list.
    for config in configs:
        if hasattr(config, "num_hidden_layers"):
            config.num_hidden_layers = len(retained_indices)
        config_types = getattr(config, "layer_types", None)
        if config_types is not None:
            config.layer_types = [config_types[i] for i in retained_indices if i < len(config_types)]

    output = Path(OUTPUT_PATH)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    AutoTokenizer.from_pretrained(MODEL_PATH).save_pretrained(output)

    actual_percentage = remove_count / layer_count * 100
    removed_summary = ", ".join(
        f"{index} (score {scores[index]:.6g})" for index in sorted(removed_indices)
    )
    print(f"Saved layer-pruned model to: {output.resolve()}")
    print(
        f"Removed {remove_count} of {layer_count} decoder blocks "
        f"({actual_percentage:.1f}%, requested {PRUNING_PERCENTAGE}%)."
    )
    print(f"Removed block indices (zero-based): {removed_summary}")


if __name__ == "__main__":
    main()
