"""Part 7: report decoder layer, attention-type, and FFN parameter counts."""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent


import os
from pathlib import Path


from torch import nn
from transformers import AutoModelForCausalLM


# Edit this setting to inspect a different local or Hub model.
MODEL_PATH = str(PROJECT_ROOT / "models" / "Prune_non_latin" / "SW3BW70")
def find_decoder(model):
    """Find the text decoder layer list, avoiding unrelated model components."""
    for module in model.modules():
        layers = getattr(module, "layers", None)
        if (
            isinstance(layers, nn.ModuleList)
            and len(layers) > 0
            and "DecoderLayer" in type(layers[0]).__name__
        ):
            return module
    # Fallback for architectures whose decoder block class has another name.
    for module in model.modules():
        layers = getattr(module, "layers", None)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            return module
    raise TypeError("Could not find a decoder layer list in this causal language model.")


def configured_layer_type(config, index):
    """Return a configured layer type when the model provides one."""
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None and index < len(layer_types):
        return str(layer_types[index]).lower()
    return ""


def has_any_attribute(module, names):
    return any(getattr(module, name, None) is not None for name in names)


def classify_layers(layers, config):
    delta_indices = []
    full_attention_indices = []
    ffn_indices = []

    for index, block in enumerate(layers):
        kind = configured_layer_type(config, index)
        linear_attention = has_any_attribute(
            block, ("linear_attn", "linear_attention", "gated_deltanet")
        ) or "deltanet" in kind or "linear_attention" in kind or "linear_attn" in kind

        if linear_attention:
            delta_indices.append(index)
        elif (
            "full_attention" in kind
            or "full_attn" in kind
            or has_any_attribute(block, ("self_attn", "attention", "attn"))
        ):
            full_attention_indices.append(index)

        if has_any_attribute(block, ("mlp", "ffn", "feed_forward", "feedforward")):
            ffn_indices.append(index)

    return full_attention_indices, delta_indices, ffn_indices


def find_ffn(block):
    """Return the feed-forward module exposed by a decoder block, if any."""
    for name in ("mlp", "ffn", "feed_forward", "feedforward"):
        module = getattr(block, name, None)
        if isinstance(module, nn.Module):
            return module
    return None


def format_indices(indices):
    return ", ".join(map(str, indices)) if indices else "none"


def main():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype="auto", low_cpu_mem_usage=True
    )
    decoder = find_decoder(model)
    layers = decoder.layers
    config = getattr(model.config, "text_config", model.config)
    full_indices, delta_indices, ffn_indices = classify_layers(layers, config)
    ffn_parameters = {
        index: sum(parameter.numel() for parameter in find_ffn(layers[index]).parameters())
        for index in ffn_indices
    }
    total_ffn_parameters = sum(ffn_parameters.values())
    ffn_neurons = {
        index: find_ffn(layers[index]).down_proj.in_features
        for index in ffn_indices
        if isinstance(getattr(find_ffn(layers[index]), "down_proj", None), nn.Linear)
    }
    total_ffn_neurons = sum(ffn_neurons.values())

    print(f"Model: {MODEL_PATH}")
    print(f"Total decoder layers: {len(layers)}")
    print(
        f"Full attention layers: {len(full_indices)}; "
        f"zero-based indices: {format_indices(full_indices)}"
    )
    print(
        f"Gated DeltaNet layers: {len(delta_indices)}; "
        f"zero-based indices: {format_indices(delta_indices)}"
    )
    print(
        f"FFN layers: {len(ffn_indices)}; "
        f"zero-based indices: {format_indices(ffn_indices)}"
    )
    print(f"Total FFN parameters: {total_ffn_parameters:,}")
    for index, parameter_count in ffn_parameters.items():
        print(f"  Layer {index}: {parameter_count:,} FFN parameters")
    print(f"Total FFN neurons across layers: {total_ffn_neurons:,}")
    for index, neuron_count in ffn_neurons.items():
        print(f"  Layer {index}: {neuron_count:,} FFN neurons")


if __name__ == "__main__":
    main()
