"""Run full MMLU, ARC-Challenge, and HellaSwag evaluations plus C4 perplexity."""

import math
import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "models/Prune_non_latin/SW3BW60X"
SEED = 1234

C4_SEQUENCES = 1000


def evaluate_c4_perplexity(model, tokenizer, device, num_batches=100, max_length=512):
    """Evaluate continuous C4 perplexity matching your WikiText evaluation protocol."""
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
                max_length=max_length
            ).to(device)
            
            input_ids = encodings.input_ids
            if input_ids.size(1) < 2:
                continue

            labels = input_ids.clone()
            outputs = model(input_ids, labels=labels)
            
            lm_loss = outputs.loss
            num_tokens = input_ids.size(1) - 1
            nlls.append(lm_loss.float().detach().cpu() * num_tokens)
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
    parser.add_argument(
        "--c4-batches",
        type=int,
        default=C4_SEQUENCES,
        help=f"Number of C4 validation sequences to score (default: {C4_SEQUENCES})",
    )
    parser.add_argument("--int4", action="store_true", help="Apply TorchAO INT4 TinyGEMM quantization")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[1/3] Loading Tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"[2/3] Loading Model into memory (INT4={args.int4})...")
    if args.int4:
        from torchao.quantization import Int4WeightOnlyConfig
        from transformers import TorchAoConfig
        int4_config = Int4WeightOnlyConfig(
            int4_packing_format="tile_packed_to_4d",
            int4_choose_qparams_algorithm="tinygemm",
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            quantization_config=TorchAoConfig(quant_type=int4_config),
            device_map={"": device},
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=dtype if device.startswith("cuda") else torch.float32,
        ).to(device)

    model.eval()

    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        parser.exit(
            2,
            "LM Evaluation Harness is required. Install it with:\n"
            "  pip install lm_eval\n"
        )

    print(
        f"\nModel: {args.model_path}\n"
        f"Batch size: {args.batch_size}\n"
        "MMLU, ARC-Challenge, and HellaSwag: full task splits\n"
        f"C4 Perplexity: Exactly {args.c4_batches} sequences\n"
    )

    # 1. Run Downstream Zero-Shot Evaluations via LM-Eval Harness (HF Backend)
    print("--- Running Downstream Zero-Shot Tasks (HF Backend) ---")
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
    )

    print("Evaluating full MMLU, ARC-Challenge, and HellaSwag...")
    downstream_results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["mmlu", "arc_challenge", "hellaswag"],
        num_fewshot=0,
        batch_size=args.batch_size,
        random_seed=SEED,
        numpy_random_seed=SEED,
        torch_random_seed=SEED,
        fewshot_random_seed=SEED,
    )

    # Print summary metrics (Safely handling both groups and results)
    print("\n================ DOWNSTREAM ZERO-SHOT RESULTS ================")
    
    # Extract MMLU Aggregate
    mmlu_data = downstream_results.get("groups", {}).get("mmlu", {}) or downstream_results.get("results", {}).get("mmlu", {})
    mmlu_acc = mmlu_data.get("acc,none", mmlu_data.get("acc_norm,none", "N/A"))
    if isinstance(mmlu_acc, float):
        print(f"  MMLU Accuracy:         {mmlu_acc * 100:.2f}%")
    else:
        print(f"  MMLU Accuracy:         {mmlu_acc}")

    # Extract ARC and HellaSwag
    for task_name in ["arc_challenge", "hellaswag"]:
        task_data = downstream_results.get("results", {}).get(task_name, {})
        acc = task_data.get("acc_norm,none", task_data.get("acc,none", "N/A"))
        if isinstance(acc, float):
            print(f"  {task_name.upper():<22} Accuracy: {acc * 100:.2f}%")
        else:
            print(f"  {task_name.upper():<22} Accuracy: {acc}")

    # Free cache before C4 streaming
    del lm
    torch.cuda.empty_cache()

    # 2. Run C4 Perplexity
    c4_ppl = evaluate_c4_perplexity(
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_batches=args.c4_batches,
    )
    print("\n[✓] Evaluation completed successfully.")


if __name__ == "__main__":
    main()
