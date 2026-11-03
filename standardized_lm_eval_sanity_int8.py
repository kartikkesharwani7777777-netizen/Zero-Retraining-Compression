"""Run reproducible downstream and C4 evaluations with TorchAO INT8 weights."""

import math
import argparse

import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
from torchao.quantization import Int8WeightOnlyConfig


MODEL_PATH = "models/Prune_non_latin/SW2BX"
SEED = 1234

MMLU_LIMIT_PER_CATEGORY = 20
OTHER_TASK_LIMIT = 1000


def evaluate_c4_perplexity(model, tokenizer, device, num_batches=100, max_length=512):
    """Evaluate continuous C4 perplexity matching the sanity evaluation protocol."""
    print("\n--- Evaluating C4 Validation Perplexity ---")
    data = load_dataset("allenai/c4", "en", split="validation", streaming=True)

    nlls = []
    total_tokens = 0
    batch_count = 0

    with torch.no_grad():
        progress = tqdm(total=num_batches, desc="C4 evaluation", unit="sequence")
        for example in data:
            text = example["text"].strip()
            if not text:
                continue

            encodings = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)

            input_ids = encodings.input_ids
            if input_ids.size(1) < 2:
                continue

            outputs = model(input_ids, labels=input_ids)
            num_tokens = input_ids.size(1) - 1
            nlls.append(outputs.loss.float().detach().cpu() * num_tokens)
            total_tokens += num_tokens

            batch_count += 1
            progress.update(1)
            if batch_count >= num_batches:
                break
        progress.close()

    ppl = math.exp(torch.stack(nlls).sum().item() / total_tokens)
    print(f"--> C4 Perplexity (first {num_batches} sequences): {ppl:.2f}")
    return ppl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--c4-batches", type=int, default=100)
    args = parser.parse_args()

    if args.c4_batches < 1:
        parser.error("--c4-batches must be at least 1")

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled NVIDIA GPU is required for TorchAO INT8 evaluation.")

    device_index = torch.cuda.current_device()
    device = f"cuda:{device_index}"
    print(f"[1/3] Loading Tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print("[2/3] Loading Model into memory (TorchAO INT8 weight-only)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            quantization_config=TorchAoConfig(quant_type=Int8WeightOnlyConfig()),
            device_map={"": device},
            trust_remote_code=True,
        )
    except ImportError as error:
        raise RuntimeError(
            "INT8 loading requires torchao and Accelerate. Install them with: "
            "python -m pip install torchao accelerate"
        ) from error
    model.eval()

    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        parser.exit(
            2,
            "LM Evaluation Harness is required. Install it with:\n"
            "  pip install lm_eval\n",
        )

    print(
        f"\nModel: {args.model_path}\n"
        "Weight quantization: INT8 weight-only (TorchAO)\n"
        f"Batch size: {args.batch_size}\n"
        f"MMLU Limit: {MMLU_LIMIT_PER_CATEGORY} examples per category\n"
        f"Other downstream limit: {OTHER_TASK_LIMIT} examples per task\n"
        f"C4 Perplexity: Exactly {args.c4_batches} sequences\n"
    )

    print("--- Running Downstream Zero-Shot Tasks (HF Backend) ---")
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)

    print("Evaluating MMLU...")
    mmlu_results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["mmlu"],
        num_fewshot=0,
        batch_size=args.batch_size,
        limit=MMLU_LIMIT_PER_CATEGORY,
        random_seed=SEED,
        numpy_random_seed=SEED,
        torch_random_seed=SEED,
        fewshot_random_seed=SEED,
    )

    torch.cuda.empty_cache()

    print("Evaluating ARC-Challenge & HellaSwag...")
    other_results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["arc_challenge", "hellaswag"],
        num_fewshot=0,
        batch_size=args.batch_size,
        limit=OTHER_TASK_LIMIT,
        random_seed=SEED,
        numpy_random_seed=SEED,
        torch_random_seed=SEED,
        fewshot_random_seed=SEED,
    )

    print("\n================ DOWNSTREAM ZERO-SHOT RESULTS ================")
    mmlu_data = mmlu_results.get("groups", {}).get("mmlu", {}) or mmlu_results.get("results", {}).get("mmlu", {})
    mmlu_acc = mmlu_data.get("acc,none", mmlu_data.get("acc_norm,none", "N/A"))
    if isinstance(mmlu_acc, float):
        print(f"  MMLU Accuracy:         {mmlu_acc * 100:.2f}%")
    else:
        print(f"  MMLU Accuracy:         {mmlu_acc}")

    for task_name in ["arc_challenge", "hellaswag"]:
        task_data = other_results.get("results", {}).get(task_name, {})
        acc = task_data.get("acc_norm,none", task_data.get("acc,none", "N/A"))
        if isinstance(acc, float):
            print(f"  {task_name.upper():<22} Accuracy: {acc * 100:.2f}%")
        else:
            print(f"  {task_name.upper():<22} Accuracy: {acc}")

    del lm
    torch.cuda.empty_cache()

    evaluate_c4_perplexity(
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_batches=args.c4_batches,
    )
    print("\n[✓] Evaluation completed successfully.")


if __name__ == "__main__":
    main()
