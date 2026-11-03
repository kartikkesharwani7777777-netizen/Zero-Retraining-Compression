"""Structurally prune Llama 3.1 FFN neurons using WANDA scores.

Uses sequential layer-by-layer GPU dispatch (the official Wanda/SparseGPT architecture)
to avoid Accelerate meta-device offload bugs and run within < 2 GB VRAM.
"""

from pathlib import Path
import torch
from torch import nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "Llama-3.1-8B"
MODEL_PATH = PROJECT_ROOT / "models" / "Llama 3.1 8B_models" / "Llama31_LayerPruned" / f"P40"
PRUNING_PERCENTAGE = 50
OUTPUT_PATH = PROJECT_ROOT / "models" / "Llama 3.1 8B_models" / "Llama31_Wanda" / f"H12W{PRUNING_PERCENTAGE}"
DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
CALIBRATION_SPLIT = "train"
NUM_CALIBRATION_BLOCKS = 32  # 32 blocks of 512 tokens = 16,384 tokens
BLOCK_SIZE = 512


def find_decoder(model):
    """Find the Llama-style module that owns the decoder layer list."""
    for module in model.modules():
        layers = getattr(module, "layers", None)
        if (
            isinstance(layers, nn.ModuleList)
            and len(layers) > 0
            and "DecoderLayer" in type(layers[0]).__name__
        ):
            return module
    raise TypeError("Could not find the Llama decoder layer list in this model.")


def prepare_calibration_data(tokenizer, num_blocks=NUM_CALIBRATION_BLOCKS, block_size=BLOCK_SIZE):
    """Concatenate and pack calibration text into dense token chunks."""
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=CALIBRATION_SPLIT)
    full_text = "\n\n".join([t for t in ds["text"] if t.strip()])
    
    tokens = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    total_tokens = num_blocks * block_size
    if len(tokens) < total_tokens:
        raise ValueError(f"Dataset has only {len(tokens)} tokens; requested {total_tokens}.")
    
    return tokens[:total_tokens].view(num_blocks, block_size)


def prune_linear_layer(linear_layer, keep_indices, dim: int):
    """Slice Linear weights in-place without re-instantiating the module."""
    # dim=0: out_features (gate_proj, up_proj)
    # dim=1: in_features  (down_proj)
    dev = linear_layer.weight.device
    idx = keep_indices.to(dev)

    with torch.no_grad():
        new_weight = linear_layer.weight.data.index_select(dim, idx).clone()
        linear_layer.weight = nn.Parameter(new_weight, requires_grad=False)
        
        if linear_layer.bias is not None:
            if dim == 0:
                new_bias = linear_layer.bias.data.index_select(0, idx).clone()
                linear_layer.bias = nn.Parameter(new_bias, requires_grad=False)
        
        if dim == 0:
            linear_layer.out_features = len(idx)
        else:
            linear_layer.in_features = len(idx)


def main():
    if not 0 <= PRUNING_PERCENTAGE < 100:
        raise ValueError("PRUNING_PERCENTAGE must be between 0 and 99.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Target acceleration device: {device}")
    print(f"Loading tokenizer & model purely to CPU from: {MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Load 100% on CPU (NO device_map="auto" to eliminate meta device / offload bugs)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    decoder = find_decoder(model)
    layers = decoder.layers
    num_layers = len(layers)

    original_width = layers[0].mlp.down_proj.in_features
    prune_count = int(original_width * (PRUNING_PERCENTAGE / 100))
    keep_count = original_width - prune_count
    print(f"Pruning width: {original_width} -> {keep_count} (-{prune_count} neurons/block, {PRUNING_PERCENTAGE}%)")

    # 2. Prepare calibration batches
    print(f"Preparing {NUM_CALIBRATION_BLOCKS} dense blocks of size {BLOCK_SIZE}...")
    batches = prepare_calibration_data(tokenizer)

    # 3. Intercept layer 0 inputs and shared attention/position arguments
    print("Capturing layer 0 inputs and attention metadata...")
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)

    inps = []
    layer_kwargs = {}

    class InputCatcher(nn.Module):
        def __init__(self, original_module):
            super().__init__()
            self.module = original_module

        def forward(self, *args, **kwargs):
            inps.append(args[0].detach().cpu())
            for k, v in kwargs.items():
                if k not in layer_kwargs:
                    layer_kwargs[k] = v
            raise StopIteration

    layers[0] = InputCatcher(layers[0])
    for batch in batches:
        try:
            with torch.inference_mode():
                model(input_ids=batch.unsqueeze(0).to(device), use_cache=False)
        except StopIteration:
            pass
    layers[0] = layers[0].module
    model.model.embed_tokens = model.model.embed_tokens.to("cpu")
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to("cpu")
    torch.cuda.empty_cache()

    # Move tensor kwargs (like position_ids, attention_mask) to GPU
    gpu_kwargs = {}
    for k, v in layer_kwargs.items():
        if torch.is_tensor(v):
            gpu_kwargs[k] = v.to(device)
        elif isinstance(v, tuple):
            gpu_kwargs[k] = tuple(t.to(device) if torch.is_tensor(t) else t for t in v)
        else:
            gpu_kwargs[k] = v

    # 4. Sequentially collect this layer's statistics, prune it, then propagate
    # calibration inputs through the pruned layer to calibrate the next layer.
    print("\nRunning sequential layer-by-layer calibration and pruning...")
    for idx, layer in enumerate(layers):
        layer = layer.to(device)
        down = layer.mlp.down_proj

        act_sq_sum = torch.zeros(down.in_features, dtype=torch.float64, device=device)

        def hook_fn(_mod, inputs):
            act = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            act_sq_sum.add_(act.double().square().sum(dim=0))

        handle = down.register_forward_pre_hook(hook_fn)

        # Collect all calibration activations with the current layer before
        # changing its weights. These outputs are deliberately discarded:
        # downstream calibration must see the pruned layer's outputs.
        with torch.inference_mode():
            for inp in inps:
                layer(inp.to(device), **gpu_kwargs)

        handle.remove()

        # Compute structured WANDA score on GPU
        input_col_l2 = act_sq_sum.sqrt().float()
        w_down_norm = down.weight.data.float().abs().sum(dim=0)
        scores = w_down_norm * input_col_l2
        keep = torch.topk(scores, k=keep_count, largest=True, sorted=False).indices.sort().values

        # Prune gate, up, down in-place
        prune_linear_layer(layer.mlp.gate_proj, keep, dim=0)
        prune_linear_layer(layer.mlp.up_proj, keep, dim=0)
        prune_linear_layer(layer.mlp.down_proj, keep, dim=1)

        # Forward again with the updated weights. Only these outputs become
        # inputs to the next layer, matching sequential Wanda calibration.
        outs = []
        with torch.inference_mode():
            for inp in inps:
                out = layer(inp.to(device), **gpu_kwargs)
                hidden = out[0] if isinstance(out, (tuple, list)) else out
                outs.append(hidden.detach().cpu())

        # Move back to CPU and pass pruned-layer outputs forward.
        layer.to("cpu")
        torch.cuda.empty_cache()
        inps = outs

        if (idx + 1) % 4 == 0 or (idx + 1) == num_layers:
            print(f"  [Layer {idx + 1:02d}/{num_layers}] Pruned & offloaded back to CPU.")

    # 5. Synchronize config
    configs = {
        id(cfg): cfg
        for cfg in (model.config, getattr(model.config, "text_config", None), decoder.config)
        if cfg is not None
    }.values()
    for cfg in configs:
        if hasattr(cfg, "intermediate_size"):
            cfg.intermediate_size = keep_count

    # 6. Save model and tokenizer
    out_dir = Path(OUTPUT_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving cleanly from CPU memory to: {out_dir}...")
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    print("[✓] Model saved successfully with zero offload conflicts!")


if __name__ == "__main__":
    main()
