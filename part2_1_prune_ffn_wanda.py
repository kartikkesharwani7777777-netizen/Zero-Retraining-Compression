"""Part 2.1: physically prune FFN neurons with Wanda activation-weighted scores."""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent


from pathlib import Path

import torch
from torch import nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# Edit these settings.
MODEL_PATH = str(PROJECT_ROOT / "models" / 'hub' / 'Qwen3.5-9B')
PRUNING_PERCENTAGE = 20
OUTPUT_PATH = str(PROJECT_ROOT / "models" / 'Prune_FFN_Wanda' / 'P20')
DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
CALIBRATION_SPLIT = "train"
CALIBRATION_EXAMPLES = 128
MAX_LENGTH = 512


def find_decoder(model):
    for module in model.modules():
        layers = getattr(module, "layers", None)
        if isinstance(layers, nn.ModuleList) and layers and "DecoderLayer" in type(layers[0]).__name__:
            return module
    raise TypeError("Could not find the text decoder in this causal language model.")


def replace_linear(linear, weight, bias):
    replacement = nn.Linear(
        weight.shape[1], weight.shape[0], bias=bias is not None,
        device=weight.device, dtype=weight.dtype,
    )
    replacement.weight = nn.Parameter(weight.contiguous(), requires_grad=linear.weight.requires_grad)
    if bias is not None:
        replacement.bias = nn.Parameter(bias.contiguous(), requires_grad=linear.bias.requires_grad)
    return replacement


def main():
    if not 0 <= PRUNING_PERCENTAGE < 100:
        raise ValueError("PRUNING_PERCENTAGE must be at least 0 and less than 100.")
    if CALIBRATION_EXAMPLES < 1 or MAX_LENGTH < 1:
        raise ValueError("Calibration example count and maximum length must be positive.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype="auto", low_cpu_mem_usage=True,device_map = "auto"
    )
    model.eval()
    decoder = find_decoder(model)
    layers = decoder.layers

    # Wanda uses activation-scaled weight magnitude: |W| * RMS(input).
    # For structured FFN neuron pruning, down_proj columns correspond to
    # intermediate neurons, so aggregate that criterion over each column.
    sums = []
    counts = []
    handles = []
    for index, block in enumerate(layers):
        down = getattr(getattr(block, "mlp", None), "down_proj", None)
        if not isinstance(down, nn.Linear):
            raise TypeError(f"Block {index} does not expose an FFN down_proj Linear layer.")
        sums.append(torch.zeros(down.in_features, dtype=torch.float64))
        counts.append(0)

        def collect_input(_module, inputs, layer_index=index):
            activation = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            sums[layer_index].add_(activation.double().square().sum(dim=0).cpu())
            counts[layer_index] += activation.shape[0]

        handles.append(down.register_forward_pre_hook(collect_input))

    try:
        data = load_dataset(DATASET_NAME, DATASET_CONFIG, split=CALIBRATION_SPLIT)
        data = data.filter(lambda row: bool(row["text"].strip()))
        data = data.select(range(min(CALIBRATION_EXAMPLES, len(data))))
        if len(data) == 0:
            raise ValueError("The selected calibration split contains no non-empty text.")
        device = next(model.parameters()).device
        with torch.inference_mode():
            for row in data:
                encoded = tokenizer(
                    row["text"], return_tensors="pt", truncation=True, max_length=MAX_LENGTH
                )
                if encoded["input_ids"].shape[1] < 2:
                    continue
                model(**{key: value.to(device) for key, value in encoded.items()}, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    if any(count == 0 for count in counts):
        raise ValueError("Calibration did not capture FFN activations for every decoder block.")

    hidden_width = layers[0].mlp.down_proj.in_features
    prune_count = int(hidden_width * PRUNING_PERCENTAGE / 100)
    if hidden_width - prune_count < 1:
        raise ValueError("Pruning must leave at least one FFN neuron.")

    for index, block in enumerate(layers):
        mlp = block.mlp
        gate, up, down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
        if not all(isinstance(proj, nn.Linear) for proj in (gate, up, down)):
            raise TypeError(f"Block {index} must expose gate_proj, up_proj, and down_proj Linear layers.")
        if gate.out_features != up.out_features or down.in_features != gate.out_features:
            raise ValueError(f"Block {index} has incompatible FFN projection dimensions.")
        if counts[index] != counts[0] or down.in_features != hidden_width:
            raise ValueError("Decoder blocks have inconsistent calibration token counts or FFN widths.")

        rms = (sums[index] / counts[index]).sqrt().to(device=down.weight.device, dtype=torch.float32)
        # Sum Wanda |W_ij| * RMS(x_j) down each intermediate-neuron column.
        score = (down.weight.detach().float().abs() * rms.unsqueeze(0)).sum(dim=0)
        keep = torch.topk(score, k=hidden_width - prune_count, largest=True, sorted=False).indices
        keep = keep.sort().values
        gate_bias = gate.bias.index_select(0, keep) if gate.bias is not None else None
        up_bias = up.bias.index_select(0, keep) if up.bias is not None else None
        down_bias = down.bias.detach().clone() if down.bias is not None else None
        mlp.gate_proj = replace_linear(gate, gate.weight.index_select(0, keep), gate_bias)
        mlp.up_proj = replace_linear(up, up.weight.index_select(0, keep), up_bias)
        mlp.down_proj = replace_linear(down, down.weight.index_select(1, keep), down_bias)

    new_width = hidden_width - prune_count
    configs = {
        id(config): config
        for config in (model.config, getattr(model.config, "text_config", None), decoder.config)
        if config is not None
    }.values()
    for config in configs:
        if hasattr(config, "intermediate_size"):
            config.intermediate_size = new_width

    output = Path(OUTPUT_PATH)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    print(f"Saved Wanda FFN-pruned model to: {output.resolve()}")
    print(
        f"Removed {prune_count} of {hidden_width} FFN neurons per decoder block "
        f"({prune_count / hidden_width * 100:.1f}%, requested {PRUNING_PERCENTAGE}%). "
        f"Calibration: {len(data)} WikiText examples, max length {MAX_LENGTH}."
    )


if __name__ == "__main__":
    main()
