"""Structurally prune attention heads from Llama 3.1 8B.

Llama 3.1 uses grouped-query attention (GQA): each KV head serves a group of
query heads. This script ranks KV groups by their Q/K/V/O projection weight
magnitudes, then removes the lowest-scoring groups from every decoder block.
"""

from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "Llama-3.1-8B"
PRUNING_PERCENTAGE = 25
OUTPUT_PATH = (
    PROJECT_ROOT / "models" / "Llama31_AttentionHeadPruned" / f"P{PRUNING_PERCENTAGE}"
)


def find_decoder(model):
    """Find the module that owns the Llama decoder block list."""
    for module in model.modules():
        layers = getattr(module, "layers", None)
        if (
            isinstance(layers, nn.ModuleList)
            and len(layers) > 0
            and "DecoderLayer" in type(layers[0]).__name__
        ):
            return module
    raise TypeError("Could not find the Llama decoder layer list in this model.")


def get_attention(block, index):
    attention = getattr(block, "self_attn", None)
    if attention is None:
        raise TypeError(f"Decoder block {index} has no self_attn module.")
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        if not isinstance(getattr(attention, name, None), nn.Linear):
            raise TypeError(f"Decoder block {index} has no Linear {name} projection.")
    return attention


def get_dimensions(attention, config):
    """Read head counts and validate that projection sizes match them."""
    head_dim = getattr(attention, "head_dim", None) or getattr(config, "head_dim", None)
    query_heads = getattr(attention, "num_heads", None) or getattr(
        attention, "num_attention_heads", None
    ) or getattr(config, "num_attention_heads", None)
    kv_heads = getattr(attention, "num_key_value_heads", None) or getattr(
        config, "num_key_value_heads", None
    )
    if not all(isinstance(value, int) and value > 0 for value in (head_dim, query_heads, kv_heads)):
        raise TypeError("Could not read valid head_dim, num_attention_heads, and num_key_value_heads.")
    if query_heads % kv_heads:
        raise ValueError("The query head count must be divisible by the KV head count.")

    q_proj, k_proj, v_proj, o_proj = (
        attention.q_proj, attention.k_proj, attention.v_proj, attention.o_proj
    )
    if q_proj.out_features != query_heads * head_dim:
        raise ValueError("q_proj output size does not match the configured query heads.")
    if k_proj.out_features != kv_heads * head_dim or v_proj.out_features != kv_heads * head_dim:
        raise ValueError("k_proj/v_proj output sizes do not match the configured KV heads.")
    if o_proj.in_features != query_heads * head_dim:
        raise ValueError("o_proj input size does not match the configured query heads.")
    return head_dim, query_heads, kv_heads


def group_importance(attention, dimensions):
    """Score each KV group using squared Q/K/V/O projection weights."""
    head_dim, query_heads, kv_heads = dimensions
    queries_per_kv = query_heads // kv_heads

    q_scores = attention.q_proj.weight.detach().float().square().reshape(
        kv_heads, queries_per_kv, head_dim, -1
    ).sum((1, 2, 3))
    k_scores = attention.k_proj.weight.detach().float().square().reshape(
        kv_heads, head_dim, -1
    ).sum((1, 2))
    v_scores = attention.v_proj.weight.detach().float().square().reshape(
        kv_heads, head_dim, -1
    ).sum((1, 2))
    o_scores = attention.o_proj.weight.detach().float().reshape(
        -1, query_heads, head_dim
    ).square().sum((0, 2)).reshape(kv_heads, queries_per_kv).sum(1)
    return q_scores + k_scores + v_scores + o_scores


def sliced_linear(linear, rows=None, columns=None):
    """Create a Linear layer containing selected rows and/or columns."""
    weight = linear.weight.detach()
    bias = linear.bias.detach() if linear.bias is not None else None
    if rows is not None:
        row_ids = rows.to(weight.device)
        weight = weight.index_select(0, row_ids)
        if bias is not None:
            bias = bias.index_select(0, row_ids.to(bias.device))
    if columns is not None:
        weight = weight.index_select(1, columns.to(weight.device))

    result = nn.Linear(
        weight.shape[1], weight.shape[0], bias=bias is not None,
        device=weight.device, dtype=weight.dtype,
    )
    result.weight = nn.Parameter(weight.contiguous(), requires_grad=linear.weight.requires_grad)
    if bias is not None:
        result.bias = nn.Parameter(bias.contiguous(), requires_grad=linear.bias.requires_grad)
    return result


def prune_attention(attention, dimensions, remove_kv_heads):
    """Remove lowest-scoring KV groups and their associated query heads."""
    head_dim, query_heads, kv_heads = dimensions
    queries_per_kv = query_heads // kv_heads
    keep_kv = group_importance(attention, dimensions).topk(
        kv_heads - remove_kv_heads, largest=True, sorted=False
    ).indices.sort().values

    query_ids = (
        keep_kv[:, None] * queries_per_kv
        + torch.arange(queries_per_kv, device=keep_kv.device)
    ).reshape(-1)
    q_rows = (
        query_ids[:, None] * head_dim
        + torch.arange(head_dim, device=keep_kv.device)
    ).reshape(-1)
    kv_rows = (
        keep_kv[:, None] * head_dim
        + torch.arange(head_dim, device=keep_kv.device)
    ).reshape(-1)
    o_columns = q_rows

    attention.q_proj = sliced_linear(attention.q_proj, rows=q_rows)
    attention.k_proj = sliced_linear(attention.k_proj, rows=kv_rows)
    attention.v_proj = sliced_linear(attention.v_proj, rows=kv_rows)
    attention.o_proj = sliced_linear(attention.o_proj, columns=o_columns)
    attention.num_heads = query_heads - remove_kv_heads * queries_per_kv
    attention.num_key_value_heads = kv_heads - remove_kv_heads
    attention.num_key_value_groups = attention.num_heads // attention.num_key_value_heads
    # Keep compatibility with implementations using this attribute name.
    if hasattr(attention, "num_attention_heads"):
        attention.num_attention_heads = attention.num_heads
    return keep_kv


def main():
    if not 0 <= PRUNING_PERCENTAGE < 100:
        raise ValueError("PRUNING_PERCENTAGE must be at least 0 and less than 100.")
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), dtype="auto", low_cpu_mem_usage=True
    )
    decoder = find_decoder(model)
    layers = [get_attention(block, index) for index, block in enumerate(decoder.layers)]
    config = getattr(model.config, "text_config", model.config)
    dimensions = [get_dimensions(attention, config) for attention in layers]
    if any(dims != dimensions[0] for dims in dimensions):
        raise ValueError("All decoder blocks must have matching attention head dimensions.")

    head_dim, query_heads, kv_heads = dimensions[0]
    remove_kv_heads = int(kv_heads * PRUNING_PERCENTAGE / 100)
    if not 0 < remove_kv_heads < kv_heads:
        raise ValueError(
            "PRUNING_PERCENTAGE must remove at least one KV head and leave one remaining."
        )

    for attention in layers:
        prune_attention(attention, (head_dim, query_heads, kv_heads), remove_kv_heads)

    new_query_heads = query_heads - remove_kv_heads * (query_heads // kv_heads)
    new_kv_heads = kv_heads - remove_kv_heads
    configs = {
        id(cfg): cfg
        for cfg in (model.config, getattr(model.config, "text_config", None), decoder.config)
        if cfg is not None
    }.values()
    for cfg in configs:
        if hasattr(cfg, "num_attention_heads"):
            cfg.num_attention_heads = new_query_heads
        if hasattr(cfg, "num_key_value_heads"):
            cfg.num_key_value_heads = new_kv_heads
        if hasattr(cfg, "head_dim"):
            cfg.head_dim = head_dim

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_PATH)
    AutoTokenizer.from_pretrained(str(MODEL_PATH)).save_pretrained(OUTPUT_PATH)

    print(f"Saved attention-pruned model to: {OUTPUT_PATH.resolve()}")
    print(
        f"Removed {remove_kv_heads}/{kv_heads} KV heads and their associated query heads "
        f"per decoder block ({PRUNING_PERCENTAGE}% requested). "
        f"Remaining heads: {new_query_heads} query, {new_kv_heads} KV, head_dim={head_dim}."
    )


if __name__ == "__main__":
    main()
