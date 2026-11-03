"""Part 2: physically prune low-importance FFN neurons in every decoder block."""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
from pathlib import Path
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Edit these settings.
MODEL_PATH = str(PROJECT_ROOT / "models" / 'hub' / 'Qwen3.5-9B')
PRUNING_PERCENTAGE = 50  # Percentage of FFN neurons to remove in each block (0-100).
OUTPUT_PATH = str(PROJECT_ROOT / "models" / 'pruned_model')
def replace_linear(linear, weight, bias):
    """Create a smaller Linear layer while preserving dtype, device, and trainability."""
    replacement = nn.Linear(
        weight.shape[1],
        weight.shape[0],
        bias=bias is not None,
        device=weight.device,
        dtype=weight.dtype,
    )
    replacement.weight = nn.Parameter(weight.contiguous(), requires_grad=linear.weight.requires_grad)
    if bias is not None:
        replacement.bias = nn.Parameter(bias.contiguous(), requires_grad=linear.bias.requires_grad)
    return replacement


def main():
    if not 0 <= PRUNING_PERCENTAGE < 100:
        raise ValueError("PRUNING_PERCENTAGE must be at least 0 and less than 100.")

    # Qwen 3.5 checkpoints are commonly stored in bfloat16. `auto` preserves
    # the checkpoint dtype instead of expanding a 9B model to float32 in RAM.
    # Low-memory loading avoids holding both the full state dict and model at
    # once while Transformers reads the checkpoint shards.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype="auto",
        low_cpu_mem_usage=True,
    )

    decoder = None
    for module in model.modules():
        layers = getattr(module, "layers", None)
        if isinstance(layers, nn.ModuleList) and layers and "DecoderLayer" in type(layers[0]).__name__:
            decoder = module
            break
    if decoder is None:
        raise TypeError("Could not find the text decoder in this causal language model.")

    layers = decoder.layers
    expected_model_width = getattr(model.config, "hidden_size", None)
    validated_projections = []
    hidden_width = None
    for layer_index, block in enumerate(layers):
        mlp = getattr(block, "mlp", None)
        gate = getattr(mlp, "gate_proj", None)
        up = getattr(mlp, "up_proj", None)
        down = getattr(mlp, "down_proj", None)
        if not all(isinstance(proj, nn.Linear) for proj in (gate, up, down)):
            raise TypeError(
                f"Block {layer_index} must have gate_proj, up_proj, and down_proj Linear layers."
            )

        if gate.in_features != up.in_features:
            raise ValueError(f"Block {layer_index}: gate and up input dimensions differ.")
        if expected_model_width is not None and gate.in_features != expected_model_width:
            raise ValueError(
                f"Block {layer_index}: gate/up input dimension {gate.in_features} does not "
                f"match model hidden_size {expected_model_width}."
            )
        if gate.out_features != up.out_features:
            raise ValueError(f"Block {layer_index}: gate and up output dimensions differ.")
        if down.in_features != gate.out_features:
            raise ValueError(
                f"Block {layer_index}: down input dimension {down.in_features} does not "
                f"match gate/up output dimension {gate.out_features}."
            )
        if expected_model_width is not None and down.out_features != expected_model_width:
            raise ValueError(
                f"Block {layer_index}: down output dimension {down.out_features} does not "
                f"match model hidden_size {expected_model_width}."
            )
        if hidden_width is None:
            hidden_width = gate.out_features
        elif gate.out_features != hidden_width:
            raise ValueError(
                f"Block {layer_index}: FFN intermediate width {gate.out_features} differs "
                f"from the other blocks' width {hidden_width}."
            )
        validated_projections.append((gate, up, down))

    if not validated_projections:
        raise ValueError("The decoder contains no blocks to prune.")
    prune_count = int(hidden_width * PRUNING_PERCENTAGE / 100)
    if hidden_width - prune_count < 1:
        raise ValueError("Pruning must leave at least one FFN neuron.")

    for block, (gate, up, down) in zip(layers, validated_projections):
        mlp = block.mlp
        # A neuron is important when its incoming gate/up weights or its outgoing
        # down-projection weights have large magnitude.
        score = (
            gate.weight.detach().float().pow(2).sum(dim=1)
            + up.weight.detach().float().pow(2).sum(dim=1)
            + down.weight.detach().float().pow(2).sum(dim=0)
        )
        if gate.bias is not None:
            score += gate.bias.detach().float().pow(2)
        if up.bias is not None:
            score += up.bias.detach().float().pow(2)

        # Keep the highest-importance neurons; the lowest scores are pruned.
        # topk avoids sorting every neuron when only a fraction must be kept.
        keep_indices = torch.topk(
            score, k=hidden_width - prune_count, largest=True, sorted=False
        ).indices
        gate_bias = gate.bias.index_select(0, keep_indices) if gate.bias is not None else None
        up_bias = up.bias.index_select(0, keep_indices) if up.bias is not None else None
        mlp.gate_proj = replace_linear(gate, gate.weight.index_select(0, keep_indices), gate_bias)
        mlp.up_proj = replace_linear(up, up.weight.index_select(0, keep_indices), up_bias)
        down_bias = down.bias.detach().clone() if down.bias is not None else None
        mlp.down_proj = replace_linear(down, down.weight.index_select(1, keep_indices), down_bias)

    new_width = hidden_width - prune_count
    # Qwen and similar architectures store the shared FFN width in their config.
    model.config.intermediate_size = new_width
    decoder.config.intermediate_size = new_width
    # Some Qwen 3.5 wrappers keep the text config nested on the root config.
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None:
        text_config.intermediate_size = new_width

    output = Path(OUTPUT_PATH)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(output)

    actual_percentage = prune_count / hidden_width * 100
    print(f"Saved structured FFN-pruned model to: {output.resolve()}")
    print(
        f"Removed {prune_count} of {hidden_width} FFN neurons per decoder block "
        f"({actual_percentage:.1f}%, requested {PRUNING_PERCENTAGE}%)."
    )


if __name__ == "__main__":
    main()
