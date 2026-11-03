"""Part 9: physically prune non-Latin tokens from a BPE causal LM vocabulary.

Tokens that decode to any character not associated with Latin script are
removed from both the tokenizer and the input/output vocabulary matrices.
Special tokens are retained. This is a vocabulary-size reduction, not a method
for removing a language from representations stored in the remaining weights.
"""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent


import os
from pathlib import Path


import json
import unicodedata
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer


# Edit these settings.
MODEL_PATH = str(PROJECT_ROOT / "models" / 'Prune_FFN_Wanda_Layerwise_2_3' / 'SW3BA50W70')
OUTPUT_PATH = str(PROJECT_ROOT / "models" / 'Prune_non_latin' / 'SW3A50BW70X')
def has_non_latin_character(text):
    """Return true when a decoded character is not associated with Latin script."""
    for character in text:
        # Unicode names identify script-specific punctuation, numerals, and
        # symbols as well as letters. Unnamed characters are retained because
        # their script cannot be inferred safely from the character alone.
        if "LATIN" not in unicodedata.name(character, ""):
            return True
    return False


def replace_rows(parameter, keep_ids):
    index = torch.tensor(keep_ids, device=parameter.device, dtype=torch.long)
    return torch.nn.Parameter(
        parameter.detach().index_select(0, index).contiguous(),
        requires_grad=parameter.requires_grad,
    )


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    if not tokenizer.is_fast or tokenizer.backend_tokenizer is None:
        raise TypeError("Part 9 requires a fast Hugging Face tokenizer.")

    original_size = len(tokenizer)
    special_ids = set(tokenizer.all_special_ids)
    keep_ids = []
    removed_ids = []

    for token_id in range(original_size):
        # Keep special/control tokens regardless of their printable spelling.
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        if token_id in special_ids or not has_non_latin_character(decoded):
            keep_ids.append(token_id)
        else:
            removed_ids.append(token_id)

    if not removed_ids:
        raise ValueError("No non-Latin vocabulary tokens were found to prune.")
    if not keep_ids:
        raise ValueError("Vocabulary pruning would leave no tokens.")

    backend_data = json.loads(tokenizer.backend_tokenizer.to_str())
    model_data = backend_data.get("model", {})
    vocab = model_data.get("vocab")
    if model_data.get("type") != "BPE" or not isinstance(vocab, dict):
        raise TypeError(
            "Part 9 currently supports fast BPE tokenizers with a token-to-ID vocab; "
            "this tokenizer uses a different vocabulary format."
        )

    # Rebuild a contiguous ID space while keeping original token ordering.
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(keep_ids)}
    new_vocab = {
        token: old_to_new[token_id]
        for token, token_id in vocab.items()
        if token_id in old_to_new
    }
    if len(set(new_vocab.values())) != len(new_vocab):
        raise ValueError(
            "Tokenizer vocabulary IDs are inconsistent; refusing to "
            "write an inconsistent checkpoint."
        )

    model_data["vocab"] = new_vocab
    backend_data["added_tokens"] = [
        {**entry, "id": old_to_new[entry["id"]]}
        for entry in backend_data.get("added_tokens", [])
        if entry.get("id") in old_to_new
    ]

    merges = model_data.get("merges")
    if isinstance(merges, list):
        # A merge may only produce a token that still exists in the reduced vocab.
        backend_data["model"]["merges"] = [
            merge
            for merge in merges
            if (
                "".join(merge)
                if isinstance(merge, list)
                else "".join(merge.split())
            )
            in new_vocab
        ]

    reduced_backend = Tokenizer.from_str(
        json.dumps(backend_data, ensure_ascii=False)
    )
    # Qwen3.5 exposes backend_tokenizer as a read-only property over _tokenizer.
    # Replace that backing field so the wrapper's metadata remains intact.
    tokenizer._tokenizer = reduced_backend

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype="auto",
        low_cpu_mem_usage=True,
    )
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()

    if input_embeddings.weight.shape[0] < original_size:
        raise ValueError(
            f"Model input vocabulary ({input_embeddings.weight.shape[0]}) is smaller "
            f"than tokenizer size ({original_size})."
        )

    input_embeddings.weight = replace_rows(input_embeddings.weight, keep_ids)
    input_bias = getattr(input_embeddings, "bias", None)
    if input_bias is not None:
        input_embeddings.bias = replace_rows(input_bias, keep_ids)

    if output_embeddings is not None and output_embeddings is not input_embeddings:
        if output_embeddings.weight.shape[0] < original_size:
            raise ValueError("Model output vocabulary is smaller than tokenizer size.")
        output_embeddings.weight = replace_rows(output_embeddings.weight, keep_ids)
        output_bias = getattr(output_embeddings, "bias", None)
        if output_bias is not None:
            output_embeddings.bias = replace_rows(output_bias, keep_ids)

    model.config.vocab_size = len(keep_ids)
    if hasattr(model.config, "text_config"):
        model.config.text_config.vocab_size = len(keep_ids)
        model.config.text_config.eos_token_id = tokenizer.eos_token_id
        model.config.text_config.bos_token_id = tokenizer.bos_token_id
        model.config.text_config.pad_token_id = tokenizer.pad_token_id
    # Vocabulary pruning renumbers token IDs. Keep all model and generation
    # configs aligned with the reduced tokenizer so IDs like EOS stay in range.
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.tie_weights()

    output = Path(OUTPUT_PATH)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)

    print(f"Saved vocabulary-pruned model to: {output.resolve()}")
    print(f"Vocabulary: {original_size} -> {len(keep_ids)} tokens")
    print(f"Removed {len(removed_ids)} tokens containing non-Latin characters.")
    print("Unicode byte fragments with unnamed characters are retained.")


if __name__ == "__main__":
    main()
