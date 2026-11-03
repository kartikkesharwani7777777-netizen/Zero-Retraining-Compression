"""Prune low-importance heads from full-attention layers; skip Gated DeltaNet."""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent


import os
from pathlib import Path


from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# Edit these settings.
MODEL_PATH = str(PROJECT_ROOT / "models" / "Prune_stratified" / "SW3B")
PRUNING_PERCENTAGE = 25
OUTPUT_PATH = str(PROJECT_ROOT / "models" / 'Prune_attentionhead' / 'SW3BA25')
def is_deltanet(block, attention):
    """Identify linear-attention (Gated DeltaNet) decoder blocks."""
    return (
        getattr(block, "linear_attn", None) is not None
        or "deltanet" in type(block).__name__.lower()
        or "deltanet" in type(attention).__name__.lower()
    )


def find_layers(model):
    """Find decoder full-attention layers and count skipped DeltaNet blocks."""
    for decoder in model.modules():
        layers = getattr(decoder, "layers", None)
        if not isinstance(layers, nn.ModuleList) or not layers:
            continue

        attention_layers, skipped = [], 0
        for index, block in enumerate(layers):
            attention = getattr(block, "self_attn", None) or getattr(block, "attn", None)
            if is_deltanet(block, attention):
                skipped += 1
                continue
            projections = [getattr(attention, name, None) for name in ("q_proj", "k_proj", "v_proj", "o_proj")]
            if not all(isinstance(projection, nn.Linear) for projection in projections):
                raise TypeError(f"Block {index} is not a recognized full-attention or Gated DeltaNet block.")
            attention_layers.append((attention, *projections))
        if attention_layers:
            return decoder, attention_layers, skipped
    raise TypeError("No full-attention decoder layers with Q/K/V/O Linear projections were found.")


def layer_dimensions(attention, q_proj, k_proj, v_proj, o_proj, config):
    """Read and check attention sizes needed to slice one layer."""
    head_dim = getattr(attention, "head_dim", None) or getattr(config, "head_dim", None)
    query_heads = (
        getattr(attention, "num_heads", None)
        or getattr(attention, "num_attention_heads", None)
        or getattr(config, "num_attention_heads", None)
    )
    kv_heads = getattr(attention, "num_key_value_heads", None) or getattr(config, "num_key_value_heads", query_heads)
    if not all(isinstance(size, int) and size > 0 for size in (head_dim, query_heads, kv_heads)):
        raise TypeError("Attention head metadata (head_dim, num_attention_heads, num_key_value_heads) is missing.")
    if query_heads % kv_heads:
        raise ValueError("Query head count must be divisible by KV head count.")

    q_width = query_heads * head_dim
    q_factor = q_proj.out_features // q_width if q_proj.out_features % q_width == 0 else 0
    if q_factor not in (1, 2):
        raise ValueError("q_proj output width does not match the declared head dimensions.")
    if (k_proj.out_features, v_proj.out_features, o_proj.in_features) != (
        kv_heads * head_dim, kv_heads * head_dim, q_width
    ):
        raise ValueError("Q/K/V/O projection widths do not match the attention head metadata.")
    return head_dim, query_heads, kv_heads, q_factor


def importance(q_proj, k_proj, v_proj, o_proj, dims):
    """Return one squared-weight importance score per KV head/query group."""
    head_dim, query_heads, kv_heads, q_factor = dims
    queries_per_kv = query_heads // kv_heads
    q = q_proj.weight.detach().float().reshape(query_heads, q_factor * head_dim, -1).square().sum((1, 2))
    o = o_proj.weight.detach().float().reshape(-1, query_heads, head_dim).square().sum((0, 2))
    k = k_proj.weight.detach().float().reshape(kv_heads, head_dim, -1).square().sum((1, 2))
    v = v_proj.weight.detach().float().reshape(kv_heads, head_dim, -1).square().sum((1, 2))
    grouped_q = q.reshape(kv_heads, queries_per_kv).sum(1)
    grouped_o = o.reshape(kv_heads, queries_per_kv).sum(1)
    return k + v + grouped_q + grouped_o


def smaller_linear(linear, rows=None, columns=None):
    """Copy selected rows and columns into a Linear layer of the right size."""
    weight = linear.weight.detach()
    bias = linear.bias.detach() if linear.bias is not None else None
    if rows is not None:
        rows = rows.to(weight.device)
        weight = weight.index_select(0, rows)
        if bias is not None:
            bias = bias.index_select(0, rows.to(bias.device))
    if columns is not None:
        weight = weight.index_select(1, columns.to(weight.device))
    result = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None,
                       device=weight.device, dtype=weight.dtype)
    result.weight = nn.Parameter(weight.contiguous(), requires_grad=linear.weight.requires_grad)
    if bias is not None:
        result.bias = nn.Parameter(bias.contiguous(), requires_grad=linear.bias.requires_grad)
    return result


def prune_layer(layer, dims, remove_kv):
    """Remove the lowest-scoring KV groups and their associated query heads."""
    attention, q_proj, k_proj, v_proj, o_proj = layer
    head_dim, query_heads, kv_heads, q_factor = dims
    queries_per_kv = query_heads // kv_heads
    keep_kv = importance(q_proj, k_proj, v_proj, o_proj, dims).topk(
        kv_heads - remove_kv, largest=True, sorted=False
    ).indices.sort().values
    query_ids = (keep_kv[:, None] * queries_per_kv + torch.arange(queries_per_kv)).flatten()
    q_rows = (query_ids[:, None] * (q_factor * head_dim) + torch.arange(q_factor * head_dim)).flatten()
    kv_rows = (keep_kv[:, None] * head_dim + torch.arange(head_dim, device=keep_kv.device)).flatten()
    o_columns = (query_ids[:, None] * head_dim + torch.arange(head_dim)).flatten()

    attention.q_proj = smaller_linear(q_proj, rows=q_rows)
    attention.k_proj = smaller_linear(k_proj, rows=kv_rows)
    attention.v_proj = smaller_linear(v_proj, rows=kv_rows)
    attention.o_proj = smaller_linear(o_proj, columns=o_columns)
    attention.num_heads = query_heads - remove_kv * queries_per_kv
    if hasattr(attention, "num_attention_heads"):
        attention.num_attention_heads = attention.num_heads
    attention.num_key_value_heads = kv_heads - remove_kv
    if hasattr(attention, "num_key_value_groups"):
        attention.num_key_value_groups = attention.num_heads // attention.num_key_value_heads


def main():
    if not 0 <= PRUNING_PERCENTAGE < 100:
        raise ValueError("PRUNING_PERCENTAGE must be at least 0 and less than 100.")

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype="auto", low_cpu_mem_usage=True)
    decoder, layers, skipped = find_layers(model)
    config = getattr(model.config, "text_config", model.config)
    dims = [layer_dimensions(*layer, config) for layer in layers]
    if any(dim != dims[0] for dim in dims):
        raise ValueError("All full-attention layers must share head dimensions for model-config saving.")
    head_dim, query_heads, kv_heads, _ = dims[0]
    configs = (model.config, getattr(model.config, "text_config", None), decoder.config)
    if not any(c is not None and getattr(c, "head_dim", None) for c in configs):
        raise TypeError("An explicit head_dim is required to reload the pruned model correctly.")

    remove_kv = int(kv_heads * PRUNING_PERCENTAGE / 100)
    if not 0 < remove_kv < kv_heads:
        raise ValueError("The percentage must remove at least one KV head group and leave one remaining.")
    for layer, layer_dims in zip(layers, dims):
        prune_layer(layer, layer_dims, remove_kv)

    new_queries, new_kv = query_heads - remove_kv * (query_heads // kv_heads), kv_heads - remove_kv
    for c in configs:
        if c is not None:
            if hasattr(c, "num_attention_heads"):
                c.num_attention_heads = new_queries
            if hasattr(c, "num_key_value_heads"):
                c.num_key_value_heads = new_kv
            if hasattr(c, "head_dim"):
                c.head_dim = head_dim

    output = Path(OUTPUT_PATH)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    AutoTokenizer.from_pretrained(MODEL_PATH).save_pretrained(output)
    print(f"Saved pruned model to: {output.resolve()}")
    print(f"Removed {remove_kv}/{kv_heads} KV groups ({100 * remove_kv / kv_heads:.1f}%) per full-attention layer; "
          f"{new_queries} query heads and {new_kv} KV heads remain. Skipped {skipped} Gated DeltaNet layers.")


if __name__ == "__main__":
    main()
