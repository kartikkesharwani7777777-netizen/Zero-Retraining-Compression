"""Measure WikiText-2 test perplexity with TorchAO INT8 weight-only quantization."""

import argparse
import math
from pathlib import Path
import threading
import time

from datasets import load_dataset
import pynvml
import torch
from torchao.quantization import Int8WeightOnlyConfig
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = str(PROJECT_ROOT / "models" / "Prune_non_latin" / "SW2BX")
DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_SPLIT = "test"
MAX_LENGTH = 2048
STRIDE = 2048
BATCH_SIZE = 4  # Adjust based on your GPU VRAM (4 or 8 is usually ideal)


class NvmlMemoryPeak:
  """Sample total device memory use through NVML during evaluation."""

  def __init__(self, device_index, interval_seconds=0.05):
    self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    self.interval_seconds = interval_seconds
    self.peak_bytes = 0
    self.running = False
    self.thread = None

  def _sample(self):
    while self.running:
      self.peak_bytes = max(
          self.peak_bytes, pynvml.nvmlDeviceGetMemoryInfo(self.handle).used
      )
      time.sleep(self.interval_seconds)

  def start(self):
    self.peak_bytes = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used
    self.running = True
    self.thread = threading.Thread(target=self._sample, daemon=True)
    self.thread.start()

  def stop(self):
    self.running = False
    if self.thread is not None:
      self.thread.join()
    self.peak_bytes = max(
        self.peak_bytes, pynvml.nvmlDeviceGetMemoryInfo(self.handle).used
    )


def measure_perplexity_batched(
    model, token_ids, max_length, stride, device, pad_token_id, batch_size=4
):
  """Score tokens using batched forward passes for significantly higher throughput."""
  total_nll = 0.0
  total_scored_tokens = 0

  # Collect all window slice indices
  windows = []
  previous_end = 0
  for start in range(0, token_ids.numel(), stride):
    end = min(start + max_length, token_ids.numel())
    valid_length = end - start
    target_start = max(previous_end - start, 0)
    first_target = max(target_start, 1)
    scored = max(valid_length - first_target, 0)

    if scored > 0:
      windows.append((start, end, valid_length, first_target, scored))
    previous_end = end
    if end == token_ids.numel():
      break

  if not windows:
    raise RuntimeError(
        "The dataset did not contain enough tokens to score perplexity."
    )

  loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

  # Process windows in batches
  batch_starts = range(0, len(windows), batch_size)
  for b_start in tqdm(
      batch_starts,
      total=(len(windows) + batch_size - 1) // batch_size,
      desc="Evaluating perplexity",
      unit="batch",
  ):
    b_windows = windows[b_start : b_start + batch_size]
    batch_tensors = []

    for start, end, valid_length, _, _ in b_windows:
      chunk = token_ids[start:end]
      if valid_length < max_length:
        padding = chunk.new_full((max_length - valid_length,), pad_token_id)
        chunk = torch.cat((chunk, padding), dim=0)
      batch_tensors.append(chunk)

    input_batch = torch.stack(batch_tensors, dim=0).to(device)

    with torch.inference_mode():
      # use_cache=False avoids caching past KV tensors across all layers
      logits = model(input_batch, use_cache=False).logits

      for i, (_, _, valid_length, first_target, scored) in enumerate(b_windows):
        shift_logits = logits[i, first_target - 1 : valid_length - 1, :].float()
        shift_labels = input_batch[i, first_target:valid_length]

        loss = loss_fn(shift_logits, shift_labels).sum()
        total_nll += loss.item()
        total_scored_tokens += scored

  return math.exp(total_nll / total_scored_tokens), total_scored_tokens


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model-path", default=MODEL_PATH)
  parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
  parser.add_argument("--stride", type=int, default=STRIDE)
  parser.add_argument(
      "--batch-size",
      type=int,
      default=BATCH_SIZE,
      help="Batch size for perplexity evaluation",
  )
  args = parser.parse_args()

  if not torch.cuda.is_available():
    raise RuntimeError(
        "A CUDA-enabled NVIDIA GPU is required for INT8 GPU evaluation."
    )

  device_index = torch.cuda.current_device()
  device = torch.device(f"cuda:{device_index}")
  dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

  tokenizer = AutoTokenizer.from_pretrained(args.model_path)
  pad_token_id = (
      tokenizer.pad_token_id
      if tokenizer.pad_token_id is not None
      else (tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0)
  )

  print("Loading model and applying TorchAO INT8 weight-only quantization...")
  model = AutoModelForCausalLM.from_pretrained(
      args.model_path,
      torch_dtype=dtype,
      quantization_config=TorchAoConfig(quant_type=Int8WeightOnlyConfig()),
      device_map={"": str(device)},
  )
  model.eval()

  dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
  full_text = "\n".join(row["text"] for row in dataset)
  token_ids = tokenizer(
      full_text, add_special_tokens=False, return_tensors="pt"
  )["input_ids"][0]

  if token_ids.numel() < 2:
    raise RuntimeError(
        "The configured WikiText-2 test split produced fewer than two tokens."
    )

  pynvml.nvmlInit()
  meter = NvmlMemoryPeak(device_index)

  # Quick warm-up
  warmup_ids = token_ids[: min(128, token_ids.numel())].unsqueeze(0).to(device)
  with torch.inference_mode():
    model(input_ids=warmup_ids, use_cache=False)
  torch.cuda.synchronize(device_index)

  meter.start()
  try:
    start_time = time.time()
    perplexity, scored_tokens = measure_perplexity_batched(
        model,
        token_ids,
        args.max_length,
        args.stride,
        device,
        pad_token_id,
        batch_size=args.batch_size,
    )
    torch.cuda.synchronize(device_index)
    elapsed = time.time() - start_time
  finally:
    meter.stop()
    pynvml.nvmlShutdown()

  print(f"\nModel: {args.model_path}")
  print("Weight quantization: INT8 weight-only (TorchAO)")
  print(f"Dataset: {DATASET_NAME}/{DATASET_CONFIG} ({DATASET_SPLIT})")
  print(f"Device: {device}")
  print(f"Scored tokens: {scored_tokens:,}")
  print(f"WikiText-2 test perplexity: {perplexity:.4f}")
  print(f"Evaluation time: {elapsed:.2f} seconds")
  print(f"NVML peak GPU memory: {meter.peak_bytes / (1024**3):.2f} GiB")


if __name__ == "__main__":
  main()
