"""Remove non-Latin tokens from the Llama 3.1 tokenizer and vocabulary rows.

Special tokens and shared ASCII text (letters, digits, whitespace, punctuation,
and symbols) are retained. The remaining vocabulary IDs are compacted, and the
same rows are kept in the input embeddings and output projection.
"""

import json
import unicodedata
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" /  "Llama-3.1-8B"
OUTPUT_PATH = PROJECT_ROOT / "models" / "Llama31_NonLatinPruned"/"Non_latin"/"PureX"


def has_non_latin_character(text):
    """Return true if text contains a character outside Latin/shared ASCII."""
    for character in text:
        # A single byte-level BPE token may not decode as complete UTF-8; the
        # tokenizer then emits U+FFFD even though the underlying bytes are
        # shared ASCII or Latin text. Keep those ambiguous fragments.
        if character == "\ufffd":
            continue
        codepoint = ord(character)
        if codepoint < 128 and (
            character.isalnum()
            or character.isspace()
            or unicodedata.category(character).startswith(("P", "S"))
        ):
            continue
        if "LATIN" not in unicodedata.name(character, ""):
            return True
    return False


def replace_rows(parameter, keep_ids):
    indices = torch.tensor(keep_ids, device=parameter.device, dtype=torch.long)
    return torch.nn.Parameter(
        parameter.detach().index_select(0, indices).contiguous(),
        requires_grad=parameter.requires_grad,
    )


def main():
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_PATH}. Update MODEL_PATH to your local checkpoint."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), use_fast=True)
    if not tokenizer.is_fast or tokenizer.backend_tokenizer is None:
        raise TypeError("Vocabulary pruning requires a fast Hugging Face tokenizer.")

    original_size = len(tokenizer)
    special_ids = set(tokenizer.all_special_ids)
    keep_ids = []
    removed_ids = []
    for token_id in range(original_size):
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        if token_id in special_ids or not has_non_latin_character(decoded):
            keep_ids.append(token_id)
        else:
            removed_ids.append(token_id)

    if not removed_ids:
        raise ValueError("No non-Latin vocabulary tokens were found to prune.")
    if not keep_ids:
        raise ValueError("Vocabulary pruning would leave no tokens.")

    # Build a reduced fast-tokenizer backend with compact IDs and valid BPE merges.
    backend_data = json.loads(tokenizer.backend_tokenizer.to_str())
    model_data = backend_data.get("model", {})
    vocab = model_data.get("vocab")
    if model_data.get("type") != "BPE" or not isinstance(vocab, dict):
        raise TypeError("Expected Llama 3.1's fast BPE tokenizer with a token-to-ID vocabulary.")

    old_to_new = {old_id: new_id for new_id, old_id in enumerate(keep_ids)}
    reduced_special_ids = {
        name: old_to_new.get(getattr(tokenizer, name, None))
        for name in ("eos_token_id", "bos_token_id", "pad_token_id")
    }
    new_vocab = {
        token: old_to_new[token_id]
        for token, token_id in vocab.items()
        if token_id in old_to_new
    }
    if len(set(new_vocab.values())) != len(new_vocab):
        raise ValueError("Tokenizer vocabulary IDs are inconsistent; refusing to save.")
    model_data["vocab"] = new_vocab
    backend_data["added_tokens"] = [
        {**entry, "id": old_to_new[entry["id"]]}
        for entry in backend_data.get("added_tokens", [])
        if entry.get("id") in old_to_new
    ]

    merges = model_data.get("merges")
    if isinstance(merges, list):
        valid_merges = []
        for merge in merges:
            if isinstance(merge, list) and len(merge) == 2:
                left, right = merge
            elif isinstance(merge, str):
                pieces = merge.split(" ", 1)
                if len(pieces) != 2:
                    continue
                left, right = pieces
            else:
                continue
            if left in new_vocab and right in new_vocab and left + right in new_vocab:
                valid_merges.append(merge)
        model_data["merges"] = valid_merges

    tokenizer._tokenizer = Tokenizer.from_str(
        json.dumps(backend_data, ensure_ascii=False)
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), dtype="auto", low_cpu_mem_usage=True
    )
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings.weight.shape[0] < original_size:
        raise ValueError("Model input embedding rows are fewer than tokenizer token IDs.")
    input_embeddings.weight = replace_rows(input_embeddings.weight, keep_ids)
    input_bias = getattr(input_embeddings, "bias", None)
    if input_bias is not None:
        input_embeddings.bias = replace_rows(input_bias, keep_ids)

    if output_embeddings is not None and output_embeddings is not input_embeddings:
        if output_embeddings.weight.shape[0] < original_size:
            raise ValueError("Model output projection rows are fewer than tokenizer token IDs.")
        output_embeddings.weight = replace_rows(output_embeddings.weight, keep_ids)
        output_bias = getattr(output_embeddings, "bias", None)
        if output_bias is not None:
            output_embeddings.bias = replace_rows(output_bias, keep_ids)

    model.config.vocab_size = len(keep_ids)
    configs = [model.config, model.generation_config]
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None:
        configs.append(text_config)
    for config in configs:
        config.vocab_size = len(keep_ids)
        for name, token_id in reduced_special_ids.items():
            setattr(config, name, token_id)
    model.tie_weights()

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    print(f"Saved vocabulary-pruned model to: {OUTPUT_PATH.resolve()}")
    print(f"Vocabulary: {original_size} -> {len(keep_ids)} tokens")
    print(f"Removed {len(removed_ids)} tokens containing non-Latin characters.")
    print("Non-Latin punctuation, digits, symbols, and emoji are removed.")


if __name__ == "__main__":
    main()
