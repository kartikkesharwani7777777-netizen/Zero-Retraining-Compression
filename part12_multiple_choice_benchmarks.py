"""Evaluate a causal language model on ARC-Challenge, WinoGrande, and HellaSwag."""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent


import os
from pathlib import Path


import argparse

import torch
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
    """Return a shared prompt, answer continuations, and zero-based label."""
    if name == "arc_challenge":
        question = row["question"]
        choices = row["choices"]
        labels = choices["label"]
        texts = choices["text"]
        prompt = f"Question: {question}\nAnswer:"
        options = [f" {text}" for text in texts]
        answer = str(row["answerKey"])
        normalized_labels = [str(label).upper() for label in labels]
        normalized_answer = answer.upper()
        if normalized_answer in normalized_labels:
            target = normalized_labels.index(normalized_answer)
        elif normalized_answer in {"A", "B", "C", "D"}:
            target = ord(normalized_answer) - ord("A")
        elif normalized_answer in {"1", "2", "3", "4"}:
            target = int(normalized_answer) - 1
        else:
            target = -1
    elif name == "winogrande":
        before, separator, after = row["sentence"].partition("_")
        if not separator:
            raise ValueError(f"WinoGrande sentence has no blank marker: {row['sentence']!r}")
        prompt = before.rstrip()
        options = [" " + row["option1"] + after, " " + row["option2"] + after]
        target = int(row["answer"]) - 1
    else:
        prompt = row["ctx"]
        options = row["endings"]
        target = int(row["label"])
    return prompt, options, target


def score_options(model, tokenizer, prompt, options, device, max_length):
    """Score every option for one example in a single model forward pass."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    prompt_len = prompt_ids.shape[1]
    sequences, target_starts = [], []
    for option in options:
        full_ids = tokenizer(prompt + option, add_special_tokens=False, return_tensors="pt")["input_ids"]
        target_start = prompt_len
        if full_ids.shape[1] <= target_start:
            # Some tokenizers merge across the prompt/answer boundary.
            answer_ids = tokenizer(option, add_special_tokens=False, return_tensors="pt")["input_ids"]
            full_ids = torch.cat((prompt_ids, answer_ids), dim=1)
        # Keep the full answer and as much recent prompt context as fits.
        answer_length = full_ids.shape[1] - target_start
        if answer_length >= max_length:
            full_ids = full_ids[:, -max_length:]
            target_start = 0
        elif full_ids.shape[1] > max_length:
            excess = full_ids.shape[1] - max_length
            full_ids = full_ids[:, excess:]
            target_start -= excess
        sequences.append(full_ids[0])
        target_starts.append(target_start)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0
    vocab_size = model.get_input_embeddings().weight.shape[0]
    pad_id = min(max(int(pad_id), 0), vocab_size - 1)
    batch_length = max(sequence.numel() for sequence in sequences)
    input_ids = torch.full((len(sequences), batch_length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for index, sequence in enumerate(sequences):
        input_ids[index, : sequence.numel()] = sequence
        attention_mask[index, : sequence.numel()] = 1
    input_ids.clamp_(min=0, max=vocab_size - 1)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits.float()
    sums, normalized = [], []
    for index, sequence in enumerate(sequences):
        start = max(target_starts[index], 1)
        end = sequence.numel()
        positions = torch.arange(start, end, device=device)
        if positions.numel() == 0:
            positions = torch.tensor([end - 1], device=device)
        token_ids = input_ids[index, positions]
        token_log_probs = logits[index, positions - 1].log_softmax(dim=-1)
        total = token_log_probs.gather(-1, token_ids.unsqueeze(-1)).sum().item()
        sums.append(total)
        normalized.append(total / positions.numel())
    return sums, normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit per benchmark.")
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if args.max_length < 2:
        parser.error("--max-length must be at least 2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == "cuda" else None
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map={"": str(device)}, trust_remote_code=True
    )
    model.eval()
    print(f"Model: {args.model_path}\nDevice: {device}\nScoring: raw and length-normalized answer log-probabilities")

    for name in args.benchmarks:
        dataset_name, config, split = BENCHMARKS[name]
        dataset = load_dataset(dataset_name, config, split=split)
        if args.max_samples is not None:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        raw_correct = norm_correct = scored = skipped = 0
        for row in tqdm(dataset, desc=name, unit="example"):
            raw_label = row.get("answer") if name == "winogrande" else row.get("label") if name == "hellaswag" else row.get("answerKey")
            if raw_label in (None, "", "-1"):
                skipped += 1
                continue
            prompt, options, target = format_example(name, row)
            if not 0 <= target < len(options):
                skipped += 1
                continue
            raw_scores, norm_scores = score_options(
                model, tokenizer, prompt, options,
                model.get_input_embeddings().weight.device, args.max_length,
            )
            raw_correct += int(max(range(len(raw_scores)), key=raw_scores.__getitem__) == target)
            norm_correct += int(max(range(len(norm_scores)), key=norm_scores.__getitem__) == target)
            scored += 1
        raw_accuracy = raw_correct / scored if scored else float("nan")
        norm_accuracy = norm_correct / scored if scored else float("nan")
        print(
            f"{name}: acc={raw_accuracy:.4%} ({raw_correct}/{scored}), "
            f"acc_norm={norm_accuracy:.4%} ({norm_correct}/{scored}); "
            f"split={split}, skipped={skipped}"
        )


if __name__ == "__main__":
    main()
