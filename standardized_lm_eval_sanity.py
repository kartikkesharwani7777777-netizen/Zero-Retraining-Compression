"""Run reproducible MMLU, ARC-Challenge, HellaSwag, and C4 evaluations."""

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = str(PROJECT_ROOT / "models" / 'Prune_non_latin' / 'SW3BW50X')
SEED = 1234

MMLU_LIMIT_PER_CATEGORY = 20
OTHER_TASK_LIMIT = 1000


def seed_everything(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def evaluate_c4_perplexity(model, tokenizer, device, num_batches=100, max_length=512):
    """Evaluate continuous C4 perplexity across fixed-length sequences."""
    print("\n--- Evaluating C4 Validation Perplexity ---")
    data = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    
    total_loss = 0.0
    total_tokens = 0
    batch_count = 0

    model.eval()
    with torch.no_grad():
        progress = tqdm(total=num_batches, desc="C4 evaluation", unit="seq")
        for example in data:
            text = example["text"].strip()
            if not text:
                continue

            encodings = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )

            input_ids = encodings.input_ids.to(device)
            # Need at least 2 tokens to compute next-token loss
            if input_ids.size(1) < 2:
                continue

            # HF CausalLM automatically computes shifted cross-entropy
            outputs = model(input_ids, labels=input_ids)
            lm_loss = outputs.loss

            num_tokens = input_ids.size(1) - 1
            total_loss += lm_loss.item() * num_tokens
            total_tokens += num_tokens

            batch_count += 1
            progress.update(1)
            if batch_count >= num_batches:
                break
        progress.close()

    if total_tokens == 0:
        return float("nan")

    ppl = math.exp(total_loss / total_tokens)
    print(f"--> C4 Perplexity (first {num_batches} sequences, {total_tokens} tokens): {ppl:.2f}")
    return ppl


def load_quantized_or_full_model(model_path: str, int4: bool, device: str, dtype: torch.dtype):
    """Safely loads model with optional TorchAO INT4 TinyGEMM quantization."""
    if int4:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
            raise RuntimeError("TorchAO INT4 TinyGEMM requires an NVIDIA GPU with Compute Capability >= 8.0.")

        try:
            from torchao.quantization import Int4WeightOnlyConfig
            from transformers import TorchAoConfig
        except ImportError as e:
            raise ImportError("Please install `torchao` to use --int4 quantization.") from e

        int4_config = Int4WeightOnlyConfig(
            int4_packing_format="tile_packed_to_4d",
            int4_choose_qparams_algorithm="tinygemm",
        )
        quant_config = TorchAoConfig(quant_type=int4_config)

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            quantization_config=quant_config,
            device_map={"": device},
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype if "cuda" in device else torch.float32,
            trust_remote_code=True,
        ).to(device)

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--c4-batches", type=int, default=100)
    parser.add_argument("--int4", action="store_true", help="Apply TorchAO INT4 TinyGEMM quantization")
    args = parser.parse_args()

    seed_everything(SEED)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"[1/3] Loading Tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-padding is critical for causal LM evaluation batches
    tokenizer.padding_side = "left"

    print(f"[2/3] Loading Model (INT4={args.int4})...")
    model = load_quantized_or_full_model(args.model_path, args.int4, device, dtype)

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
        f"Batch size: {args.batch_size}\n"
        f"MMLU Limit: {MMLU_LIMIT_PER_CATEGORY} examples per subcategory\n"
        f"Downstream Limit: {OTHER_TASK_LIMIT} examples\n"
        f"C4 Perplexity: {args.c4_batches} sequences\n"
    )

    # Wrap model for lm-eval
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
    )

    print("--- Running Downstream Zero-Shot Tasks (MMLU) ---")
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

    print("--- Running Downstream Zero-Shot Tasks (ARC-C & HellaSwag) ---")
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
    # Extract MMLU
    mmlu_group = mmlu_results.get("groups", {}).get("mmlu", {})
    mmlu_res = mmlu_results.get("results", {}).get("mmlu", {})
    mmlu_acc = (
        mmlu_group.get("acc,none")
        or mmlu_group.get("acc_norm,none")
        or mmlu_res.get("acc,none")
        or mmlu_res.get("acc_norm,none")
        or "N/A"
    )
    if isinstance(mmlu_acc, (int, float)):
        print(f"  MMLU Accuracy:         {mmlu_acc * 100:.2f}%")
    else:
        print(f"  MMLU Accuracy:         {mmlu_acc}")

    # Extract ARC and HellaSwag
    for task_name in ["arc_challenge", "hellaswag"]:
        task_data = other_results.get("results", {}).get(task_name, {})
        acc = task_data.get("acc_norm,none", task_data.get("acc,none", "N/A"))
        if isinstance(acc, (int, float)):
            print(f"  {task_name.upper():<22} Accuracy: {acc * 100:.2f}%")
        else:
            print(f"  {task_name.upper():<22} Accuracy: {acc}")

    del lm
    torch.cuda.empty_cache()

    # 3. Evaluate C4 Perplexity
    evaluate_c4_perplexity(
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_batches=args.c4_batches,
    )
    print("\n[✓] Evaluation completed successfully.")


if __name__ == "__main__":
    main()
