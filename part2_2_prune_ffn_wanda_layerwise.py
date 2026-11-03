"""Part 2.2: prune FFN neurons with Wanda, processing decoder blocks one at a time."""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent


import os
from pathlib import Path


from pathlib import Path

import torch
from torch import nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask


# Edit these settings.
MODEL_PATH = str(PROJECT_ROOT / "models" / 'hub' / 'Qwen3.5-9B')
PRUNING_PERCENTAGE = 20
OUTPUT_PATH = str(PROJECT_ROOT / "models" / 'Prune_FFN_Wanda_Layerwise' / 'P20')
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


def prune_block(block, index, percentage, rms):
    mlp = block.mlp
    gate, up, down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
    if not all(isinstance(proj, nn.Linear) for proj in (gate, up, down)):
        raise TypeError(f"Block {index} must expose gate_proj, up_proj, and down_proj Linear layers.")
    if gate.out_features != up.out_features or down.in_features != gate.out_features:
        raise ValueError(f"Block {index} has incompatible FFN projection dimensions.")

    width = down.in_features
    prune_count = int(width * percentage / 100)
    if width - prune_count < 1:
        raise ValueError(f"Pruning block {index} must leave at least one FFN neuron.")

    # Wanda structured score: sum |down_proj weight| * RMS(input) by neuron column.
    scale = rms.to(device=down.weight.device, dtype=torch.float32)
    scores = (down.weight.detach().float().abs() * scale.unsqueeze(0)).sum(dim=0)
    keep = torch.topk(scores, k=width - prune_count, largest=True, sorted=False).indices.sort().values
    gate_bias = gate.bias.index_select(0, keep) if gate.bias is not None else None
    up_bias = up.bias.index_select(0, keep) if up.bias is not None else None
    down_bias = down.bias.detach().clone() if down.bias is not None else None
    mlp.gate_proj = replace_linear(gate, gate.weight.index_select(0, keep), gate_bias)
    mlp.up_proj = replace_linear(up, up.weight.index_select(0, keep), up_bias)
    mlp.down_proj = replace_linear(down, down.weight.index_select(1, keep), down_bias)
    return width, prune_count


def main():
    if not 0 <= PRUNING_PERCENTAGE < 100:
        raise ValueError("PRUNING_PERCENTAGE must be at least 0 and less than 100.")
    if CALIBRATION_EXAMPLES < 1 or MAX_LENGTH < 1:
        raise ValueError("Calibration example count and maximum length must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("Layerwise block processing requires a CUDA GPU.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype="auto", low_cpu_mem_usage=True
    )
    model.eval()
    decoder = find_decoder(model)
    layers = decoder.layers
    data = load_dataset(DATASET_NAME, DATASET_CONFIG, split=CALIBRATION_SPLIT)
    data = data.filter(lambda row: bool(row["text"].strip()))
    data = data.select(range(min(CALIBRATION_EXAMPLES, len(data))))
    if len(data) == 0:
        raise ValueError("The selected calibration split contains no non-empty text.")

    layer_count = len(layers)
    if layer_count == 0:
        raise ValueError("The decoder contains no blocks to prune.")
    if not all(hasattr(decoder, name) for name in ("embed_tokens", "rotary_emb")):
        raise TypeError("Layerwise calibration requires a decoder with embed_tokens and rotary_emb.")

    # Embed each example once and cache the hidden state entering block 0 on CPU.
    # The loop below invokes decoder blocks directly; it never calls model(...).
    embedding_device = torch.device("cpu")
    calibration_states = []
    with torch.inference_mode():
        for row in data:
            encoded = tokenizer(
                row["text"], return_tensors="pt", truncation=True, max_length=MAX_LENGTH
            )
            input_ids = encoded["input_ids"].to(embedding_device)
            if input_ids.shape[1] < 2:
                continue
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(embedding_device)
            hidden_states = decoder.embed_tokens(input_ids)
            # Qwen3.5 uses four-axis rotary positions; the text axes all match
            # for text-only WikiText inputs, as in its regular forward method.
            position_ids = torch.arange(
                hidden_states.shape[1], device=embedding_device
            ).view(1, 1, -1).expand(4, hidden_states.shape[0], -1)
            position_embeddings = decoder.rotary_emb(hidden_states, position_ids)
            mask_kwargs = {
                "config": decoder.config,
                "inputs_embeds": hidden_states,
                "attention_mask": attention_mask,
                "past_key_values": None,
                "position_ids": None,
            }
            attention_masks = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }
            calibration_states.append({
                "hidden_states": hidden_states.cpu(),
                "position_embeddings": tuple(value.cpu() for value in position_embeddings),
                "position_ids": None,
                "attention_masks": {
                    key: value.cpu() if isinstance(value, torch.Tensor) else value
                    for key, value in attention_masks.items()
                },
            })
    if not calibration_states:
        raise ValueError("The selected calibration split contains no examples with at least two tokens.")

    first_width = layers[0].mlp.down_proj.in_features
    expected_removed = int(first_width * PRUNING_PERCENTAGE / 100)

    for index, block in enumerate(layers):
        block = block.to("cuda")
        down = getattr(getattr(block, "mlp", None), "down_proj", None)
        if not isinstance(down, nn.Linear):
            raise TypeError(f"Block {index} does not expose an FFN down_proj Linear layer.")
        if down.in_features != first_width:
            raise ValueError(f"Block {index} has a different FFN width from block 0.")

        activation_sum = torch.zeros(down.in_features, dtype=torch.float64)
        token_count = 0
        layer_type = decoder.config.layer_types[index]
        if layer_type not in ("full_attention", "linear_attention"):
            raise ValueError(f"Unsupported attention type {layer_type!r} in block {index}.")

        def collect_input(_module, inputs):
            nonlocal token_count
            activation = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            activation_sum.add_(activation.double().square().sum(dim=0).cpu())
            token_count += activation.shape[0]

        handle = down.register_forward_pre_hook(collect_input)
        try:
            with torch.inference_mode():
                block_device = torch.device("cuda")
                for state in calibration_states:
                    hidden_states = state["hidden_states"].to(block_device)
                    position_embeddings = tuple(
                        value.to(block_device) for value in state["position_embeddings"]
                    )
                    attention_mask = state["attention_masks"][layer_type]
                    if isinstance(attention_mask, torch.Tensor):
                        attention_mask = attention_mask.to(block_device)
                    block(
                        hidden_states,
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                        position_ids=state["position_ids"],
                        past_key_values=None,
                        use_cache=False,
                    )
        finally:
            handle.remove()
        if token_count == 0:
            raise ValueError(f"Calibration did not capture activations for block {index}.")

        rms = (activation_sum / token_count).sqrt()
        width, removed = prune_block(block, index, PRUNING_PERCENTAGE, rms)
        if removed != expected_removed:
            raise ValueError(f"Block {index} produced an unexpected prune count.")
        del activation_sum, rms

        # Recompute this block's outputs with its newly pruned FFN so the next
        # block receives hidden states from the partially pruned model.
        with torch.inference_mode():
            block_device = torch.device("cuda")
            for state in calibration_states:
                hidden_states = state["hidden_states"].to(block_device)
                position_embeddings = tuple(
                    value.to(block_device) for value in state["position_embeddings"]
                )
                attention_mask = state["attention_masks"][layer_type]
                if isinstance(attention_mask, torch.Tensor):
                    attention_mask = attention_mask.to(block_device)
                out = block(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    position_ids=state["position_ids"],
                    past_key_values=None,
                    use_cache=False,
                )
                out = out[0] if isinstance(out, tuple) else out
                state["hidden_states"] = out.cpu()
        print(f"Pruned block {index + 1}/{layer_count}: removed {removed} of {width} FFN neurons.")
        block = block.to("cpu")
        torch.cuda.empty_cache()

    del calibration_states

    new_width = first_width - expected_removed
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
    print(f"Saved layerwise Wanda FFN-pruned model to: {output.resolve()}")
    print(
        f"Removed {expected_removed} of {first_width} FFN neurons per block "
        f"({expected_removed / first_width * 100:.1f}%, requested {PRUNING_PERCENTAGE}%). "
        f"Calibration: {len(data)} WikiText rows ({token_count} tokens in the last block), "
        f"max length {MAX_LENGTH}."
    )


if __name__ == "__main__":
    main()
