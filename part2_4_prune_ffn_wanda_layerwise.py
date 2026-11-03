"""Part 2.4: layerwise Wanda pruning with protected FFN neuron indices."""

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
#MODEL_PATH = 'C:\\Users\\LabAdmin\\Desktop\\Statistics_Project_2\\models\\hub\\Qwen3.5-9B'
MODEL_PATH = str(PROJECT_ROOT / "models" / 'Prune_non_latin' / 'SW3BX')
#MODEL_PATH = 'C:\\Users\\LabAdmin\\Desktop\\Statistics_Project_2\\models\\Prune_attentionhead\\SW3BA25'
PRUNING_PERCENTAGE = 60
OUTPUT_PATH = str(PROJECT_ROOT / "models" / 'Prune_FFN_Wanda_Layerwise_2_4' / 'SW3BXW60_V2')
CALIBRATION_EXAMPLES = 128
MAX_LENGTH = 512
# Zero-based FFN neuron positions to retain in every decoder block.
PROTECTED_INDICES = [0,1,2,3,16,17,18,19]


def find_decoder(model):
    for module in model.modules():
        layers = getattr(module, "layers", None)
        if isinstance(layers, nn.ModuleList) and layers and "DecoderLayer" in type(layers[0]).__name__:
            return module
    raise TypeError("Could not find the text decoder.")


def shrink(linear, weight, bias=None):
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None,
                      device=weight.device, dtype=weight.dtype)
    layer.weight = nn.Parameter(weight.contiguous(), requires_grad=linear.weight.requires_grad)
    if bias is not None:
        layer.bias = nn.Parameter(bias.contiguous(), requires_grad=linear.bias.requires_grad)
    return layer


def prune(block, rms, count, protected_indices):
    mlp = block.mlp
    gate, up, down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
    width = down.in_features
    if not all(isinstance(layer, nn.Linear) for layer in (gate, up, down)) or gate.out_features != width:
        raise TypeError("Expected matching gate_proj, up_proj, and down_proj Linear layers.")

    score = (down.weight.detach().float().abs() * rms.to(down.weight.device).unsqueeze(0)).sum(0)
    protected = torch.tensor(sorted(protected_indices), device=score.device, dtype=torch.long)
    eligible_mask = torch.ones(width, device=score.device, dtype=torch.bool)
    eligible_mask[protected] = False
    eligible = torch.arange(width, device=score.device)[eligible_mask]
    if count > eligible.numel():
        raise ValueError(
            f"Cannot remove {count} neurons while protecting {protected.numel()} of {width}."
        )
    removed = eligible[torch.topk(score[eligible], count, largest=False, sorted=False).indices]
    keep_mask = torch.ones(width, device=score.device, dtype=torch.bool)
    keep_mask[removed] = False
    keep = torch.arange(width, device=score.device)[keep_mask]
    mlp.gate_proj = shrink(gate, gate.weight[keep], gate.bias[keep] if gate.bias is not None else None)
    mlp.up_proj = shrink(up, up.weight[keep], up.bias[keep] if up.bias is not None else None)
    mlp.down_proj = shrink(down, down.weight[:, keep], down.bias)


def main():
    if not 0 <= PRUNING_PERCENTAGE < 100 or CALIBRATION_EXAMPLES < 1 or MAX_LENGTH < 1:
        raise ValueError("Use pruning in [0, 100) and positive calibration limits.")
    if not torch.cuda.is_available():
        raise RuntimeError("This layerwise script requires a CUDA GPU.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype="auto", low_cpu_mem_usage=True).eval()
    decoder = find_decoder(model)
    layers = decoder.layers
    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    data = data.filter(lambda row: bool(row["text"].strip())).select(range(min(CALIBRATION_EXAMPLES, len(data))))
    if not len(data) or not layers:
        raise ValueError("Calibration data and decoder layers must be non-empty.")

    # Cache only block inputs on CPU; later blocks see outputs of already-pruned blocks.
    states = []
    with torch.inference_mode():
        for row in data:
            ids = tokenizer(row["text"], return_tensors="pt", truncation=True,
                            max_length=MAX_LENGTH)["input_ids"]
            if ids.shape[1] < 2:
                continue
            hidden = decoder.embed_tokens(ids)
            positions = torch.arange(ids.shape[1]).view(1, 1, -1).expand(3, 1, -1)
            # CPU half-precision trig/complex operations are not supported by
            # every PyTorch backend. Compute RoPE in float32 and cast it to
            # the hidden-state dtype when the block runs on CUDA.
            rotary = decoder.rotary_emb(hidden.float(), positions)
            kwargs = dict(config=decoder.config, inputs_embeds=hidden, attention_mask=None,
                          past_key_values=None, position_ids=None)
            masks = {"full_attention": create_causal_mask(**kwargs),
                     "linear_attention": create_recurrent_attention_mask(**kwargs)}
            states.append((hidden, rotary, masks))
    if not states:
        raise ValueError("No calibration examples contained at least two tokens.")

    width = layers[0].mlp.down_proj.in_features
    protected_indices = set(PROTECTED_INDICES)
    if any(not isinstance(index, int) or isinstance(index, bool) for index in protected_indices):
        raise TypeError("PROTECTED_INDICES must contain integer neuron indices.")
    if any(index < 0 or index >= width for index in protected_indices):
        raise ValueError(f"PROTECTED_INDICES must be between 0 and {width - 1}.")
    count = int(width * PRUNING_PERCENTAGE / 100)
    if width - count < 1:
        raise ValueError("Pruning must leave at least one FFN neuron.")
    if count > width - len(protected_indices):
        raise ValueError("The requested pruning count exceeds the unprotected neuron count.")
    types = decoder.config.layer_types
    for i, block in enumerate(layers):
        kind = types[i]
        if kind not in ("full_attention", "linear_attention"):
            raise ValueError(f"Unsupported attention type {kind!r} in block {i}.")
        block.cuda()
        total = torch.zeros(width, device="cuda", dtype=torch.float32)

        def capture(_module, inputs):
            # Accumulate on-device: copying every FFN activation to CPU here
            # would move gigabytes over PCIe for a typical calibration run.
            x = inputs[0].detach().float().reshape(-1, width)
            total.add_(x.square().sum(0))

        handle = block.mlp.down_proj.register_forward_pre_hook(capture)
        try:
            with torch.inference_mode():
                for hidden, rotary, masks in states:
                    mask = masks[kind]
                    block(hidden.cuda(), position_embeddings=tuple(
                              x.to(device="cuda", dtype=hidden.dtype) for x in rotary
                          ),
                          attention_mask=mask.cuda() if isinstance(mask, torch.Tensor) else mask,
                          use_cache=False)
        finally:
            handle.remove()
        tokens = sum(hidden.shape[1] for hidden, _, _ in states)
        rms = (total / tokens).sqrt().cpu()
        prune(block, rms, count, protected_indices)

        with torch.inference_mode():
            for j, (hidden, rotary, masks) in enumerate(states):
                mask = masks[kind]
                out = block(hidden.cuda(), position_embeddings=tuple(
                                x.to(device="cuda", dtype=hidden.dtype) for x in rotary
                            ),
                            attention_mask=mask.cuda() if isinstance(mask, torch.Tensor) else mask,
                            use_cache=False)
                hidden_out = out[0] if isinstance(out, (tuple, list)) else out
                states[j] = (hidden_out.cpu(), rotary, masks)
        print(f"Pruned block {i + 1}/{len(layers)}: removed {count}/{width} neurons; "
              f"protected {len(protected_indices)} indices.")
        block.cpu()
        torch.cuda.empty_cache()

    for config in {id(c): c for c in (model.config, getattr(model.config, "text_config", None), decoder.config) if c}.values():
        if hasattr(config, "intermediate_size"):
            config.intermediate_size = width - count
    output = Path(OUTPUT_PATH)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    print(f"Saved model to {output.resolve()}")


if __name__ == "__main__":
    main()
