"""Structurally prune decoder blocks from Llama 3.1 8B.

Blocks are ranked by mean squared parameter magnitude, and the lowest-scoring
blocks are removed. Adjust PRUNING_PERCENTAGE before running.
"""

from pathlib import Path

from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" /  "Llama-3.1-8B"
PRUNING_PERCENTAGE = 60
PROTECTED_FIRST_LAYERS = 4
PROTECTED_LAST_LAYERS = 3
OUTPUT_PATH = (
    PROJECT_ROOT / "models" / "Llama 3.1 8B_models" / "Llama31_LayerPruned" / f"P{PRUNING_PERCENTAGE}"
)


def find_decoder(model):
    """Find the module containing the Llama decoder block list."""
    for module in model.modules():
        layers = getattr(module, "layers", None)
        if (
            isinstance(layers, nn.ModuleList)
            and len(layers) > 0
            and "DecoderLayer" in type(layers[0]).__name__
        ):
            return module
    raise TypeError("Could not find the Llama decoder layer list in this model.")


def block_importance(block):
    """Return mean squared parameter magnitude for one block."""
    squared_sum = 0.0
    parameter_count = 0
    for parameter in block.parameters():
        values = parameter.detach().float()
        squared_sum += values.square().sum().item()
        parameter_count += values.numel()
    if parameter_count == 0:
        raise ValueError("A decoder block has no parameters to score.")
    return squared_sum / parameter_count


def main():
    if not 0 <= PRUNING_PERCENTAGE < 100:
        raise ValueError("PRUNING_PERCENTAGE must be at least 0 and less than 100.")
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_PATH}\n"
            "Place the Llama 3.1 8B checkpoint at models/hub/Meta-Llama-3.1-8B "
            "or update MODEL_PATH."
        )

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), dtype="auto", low_cpu_mem_usage=True
    )
    decoder = find_decoder(model)
    layers = decoder.layers
    layer_count = len(layers)
    protected_count = PROTECTED_FIRST_LAYERS + PROTECTED_LAST_LAYERS
    if layer_count <= protected_count:
        raise ValueError(
            f"Cannot protect the first {PROTECTED_FIRST_LAYERS} and last "
            f"{PROTECTED_LAST_LAYERS} layers: model has only {layer_count} layers."
        )

    # Keep the requested percentage relative to the full stack, but only rank
    # the unprotected middle layers as pruning candidates.
    remove_count = int(layer_count * PRUNING_PERCENTAGE / 100)
    candidate_indices = range(PROTECTED_FIRST_LAYERS, layer_count - PROTECTED_LAST_LAYERS)
    if remove_count > len(candidate_indices):
        raise ValueError(
            f"Requested pruning removes {remove_count} layers, but only "
            f"{len(candidate_indices)} middle layers are eligible."
        )
    if remove_count == 0:
        raise ValueError(
            "PRUNING_PERCENTAGE rounds down to zero removable decoder blocks."
        )
    if remove_count >= layer_count:
        raise ValueError("Pruning must leave at least one decoder block.")

    scores = {i: block_importance(layers[i]) for i in candidate_indices}
    removed_indices = set(
        sorted(candidate_indices, key=scores.__getitem__)[:remove_count]
    )
    kept_indices = [i for i in range(layer_count) if i not in removed_indices]
    decoder.layers = nn.ModuleList(layers[i] for i in kept_indices)

    # Keep model and nested text configs consistent with the shortened stack.
    configs = {
        id(config): config
        for config in (
            model.config,
            getattr(model.config, "text_config", None),
            getattr(decoder, "config", None),
        )
        if config is not None
    }.values()
    for config in configs:
        if hasattr(config, "num_hidden_layers"):
            config.num_hidden_layers = len(kept_indices)
        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None:
            config.layer_types = [layer_types[i] for i in kept_indices]

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_PATH)
    AutoTokenizer.from_pretrained(str(MODEL_PATH)).save_pretrained(OUTPUT_PATH)

    print(f"Saved layer-pruned model to: {OUTPUT_PATH.resolve()}")
    print(
        f"Removed {remove_count} of {layer_count} decoder blocks "
        f"({remove_count / layer_count * 100:.1f}%, requested "
        f"{PRUNING_PERCENTAGE}%)."
    )
    print(
        f"Protected zero-based indices: 0-{PROTECTED_FIRST_LAYERS - 1} and "
        f"{layer_count - PROTECTED_LAST_LAYERS}-{layer_count - 1}."
    )
    print("Removed zero-based block indices:", ", ".join(map(str, sorted(removed_indices))))


if __name__ == "__main__":
    main()
