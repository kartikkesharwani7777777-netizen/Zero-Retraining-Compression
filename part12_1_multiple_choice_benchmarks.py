"""Fast continuation-likelihood benchmarks for base causal language models."""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

import argparse

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = str(PROJECT_ROOT / "models" / 'Prune_non_latin' / 'SW3BW60X')
BENCHMARKS = {
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test"),
    "winogrande": ("allenai/winogrande", "winogrande_xl", "validation"),
    "hellaswag": ("Rowan/hellaswag", None, "validation"),
}


def format_example(name, row):
    """Represent each answer as a text continuation of its context."""
    if name == "arc_challenge": 
        choices = row["choices"]
        labels = [str(label).upper() for label in choices["label"]]
        answer = str(row["answerKey"]).upper()
        if answer in labels:
            target = labels.index(answer)
        elif answer in {"A", "B", "C", "D"}:
            target = ord(answer) - ord("A")
        elif answer in {"1", "2", "3", "4"}:
            target = int(answer) - 1
        else:
            target = -1
        return f"{row['question'].rstrip()}\n", [" " + text for text in choices["text"]], target

    if name == "winogrande":
        before, separator, after = row["sentence"].partition("_")
        if not separator:
            raise ValueError(f"WinoGrande sentence has no blank marker: {row['sentence']!r}")
        options = [" " + row[f"option{i}"] + after for i in (1, 2)]
        return before.rstrip(), options, int(row["answer"]) - 1

    return row["ctx"], row["endings"], int(row["label"])


def continuation_ids(tokenizer, prompt, continuation, max_length):
    """Return ids and the first token position belonging to the continuation.

    The common token prefix handles tokenizers that merge across the text
    boundary. Long prompts are truncated from the left while retaining answers.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + continuation, add_special_tokens=False)["input_ids"]
    common = 0
    for left, right in zip(prompt_ids, full_ids):
        if left != right:
            break
        common += 1
    start = common
    if len(full_ids) > max_length:
        if len(full_ids) - start >= max_length:
            full_ids = full_ids[-max_length:]
            start = 0
        else:
            trim = len(full_ids) - max_length
            full_ids = full_ids[trim:]
            start -= trim
    return full_ids, start


def score_batch(model, tokenizer, examples, device, max_length):
    """Score a batch of examples with one padded forward pass."""
    candidates = []
    for prompt, options in examples:
        candidates.extend(
            continuation_ids(tokenizer, prompt, option, max_length)
            for option in options
        )

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0
    width = max(len(ids) for ids, _ in candidates)
    input_ids = torch.full((len(candidates), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for i, (ids, _) in enumerate(candidates):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        attention_mask[i, :len(ids)] = 1

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    # Build all candidate target positions together. Causal prediction at token
    # position p uses the hidden state at p - 1, so only those vectors need the
    # vocabulary projection.
    candidate_indices, prediction_positions, target_ids = [], [], []
    token_counts = []
    candidate_to_example = []
    candidate_index = 0
    for example_index, (_, options) in enumerate(examples):
        for _ in options:
            ids, start = candidates[candidate_index]
            if len(ids) < 2:
                raise ValueError("A multiple-choice candidate must contain at least two tokens.")
            first = min(max(start, 1), len(ids) - 1)
            positions = list(range(first, len(ids)))
            candidate_indices.extend([candidate_index] * len(positions))
            prediction_positions.extend(position - 1 for position in positions)
            target_ids.extend(ids[position] for position in positions)
            token_counts.append(len(positions))
            candidate_to_example.append(example_index)
            candidate_index += 1

    candidate_indices = torch.tensor(candidate_indices, device=device)
    prediction_positions = torch.tensor(prediction_positions, device=device)
    target_ids = torch.tensor(target_ids, device=device)

    with torch.inference_mode():
        backbone_output = model.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden_states = (
            backbone_output.last_hidden_state
            if hasattr(backbone_output, "last_hidden_state")
            else backbone_output[0]
        )
        selected_hidden = hidden_states[candidate_indices, prediction_positions]
        logits = model.get_output_embeddings()(selected_hidden).float()

    # One cross-entropy call scores every selected continuation token, then
    # scores are aggregated per candidate on-device.
    token_nll = F.cross_entropy(
        logits, target_ids, reduction="none"
    )
    candidate_scores = torch.zeros(len(candidates), device=device, dtype=torch.float32)
    candidate_scores.scatter_add_(0, candidate_indices, -token_nll)
    candidate_counts = torch.tensor(token_counts, device=device, dtype=torch.float32)
    normalized_scores = candidate_scores / candidate_counts

    # Transfer only the final scalar scores to the host, once per batch.
    final_scores = torch.stack((candidate_scores, normalized_scores), dim=1).tolist()
    raw_values = [pair[0] for pair in final_scores]
    norm_values = [pair[1] for pair in final_scores]
    scores, normalized, offset = [], [], 0
    for _, options in examples:
        count = len(options)
        scores.append(raw_values[offset:offset + count])
        normalized.append(norm_values[offset:offset + count])
        offset += count
    return scores, normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit per benchmark.")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1, help="Questions per forward pass; all choices are batched.")
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if args.max_length < 2 or args.batch_size < 1:
        parser.error("--max-length must be at least 2 and --batch-size must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = None
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map={"": str(device)}, trust_remote_code=True
    )
    model.eval()
    input_device = model.get_input_embeddings().weight.device
    print(f"Model: {args.model_path}\nDevice: {device}\nScoring: base-model continuation log-probability")

    for name in args.benchmarks:
        dataset_name, config, split = BENCHMARKS[name]
        dataset = load_dataset(dataset_name, config, split=split)
        if args.max_samples is not None:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        examples = []
        raw_correct = norm_correct = scored = skipped = 0

        def evaluate_pending():
            nonlocal raw_correct, norm_correct, scored
            raw_scores, norm_scores = score_batch(
                model, tokenizer, examples, input_device, args.max_length
            )
            for raw, norm, (_, _, target) in zip(raw_scores, norm_scores, pending_targets):
                raw_correct += int(max(range(len(raw)), key=raw.__getitem__) == target)
                norm_correct += int(max(range(len(norm)), key=norm.__getitem__) == target)
                scored += 1
            examples.clear()
            pending_targets.clear()

        pending_targets = []
        for row in tqdm(dataset, desc=name, unit="example"):
            raw_label = row.get("answer") if name == "winogrande" else row.get("label") if name == "hellaswag" else row.get("answerKey")
            if raw_label in (None, "", "-1"):
                skipped += 1
                continue
            prompt, options, target = format_example(name, row)
            if not 0 <= target < len(options):
                skipped += 1
                continue
            examples.append((prompt, options))
            pending_targets.append((prompt, options, target))
            if len(examples) == args.batch_size:
                evaluate_pending()
        if examples:
            evaluate_pending()

        print(
            f"{name}: acc={raw_correct / scored:.4%} ({raw_correct}/{scored}), "
            f"acc_norm={norm_correct / scored:.4%} ({norm_correct}/{scored}); "
            f"split={split}, skipped={skipped}"
        )


if __name__ == "__main__":
    main()
