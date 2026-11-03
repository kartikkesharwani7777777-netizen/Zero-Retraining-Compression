"""Part 3: fine-tune the pruned model on WikiText-2."""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# Edit these settings.
MODEL_PATH = str(PROJECT_ROOT / "models" / 'pruned_model')
OUTPUT_PATH = str(PROJECT_ROOT / "models" / 'finetuned_model')
DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
MAX_TRAIN_EXAMPLES = 6  # Set to None to use the full split.
MAX_EVAL_EXAMPLES = 2
EPOCHS = 1
LEARNING_RATE = 5e-5
BATCH_SIZE = 2
MAX_LENGTH = 512


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.config.pad_token_id = tokenizer.pad_token_id

    raw = load_dataset(DATASET_NAME, DATASET_CONFIG)
    train_data = raw["train"].filter(lambda row: bool(row["text"].strip()))
    eval_data = raw["validation"].filter(lambda row: bool(row["text"].strip()))
    if MAX_TRAIN_EXAMPLES is not None:
        train_data = train_data.select(range(min(MAX_TRAIN_EXAMPLES, len(train_data))))
    if MAX_EVAL_EXAMPLES is not None:
        eval_data = eval_data.select(range(min(MAX_EVAL_EXAMPLES, len(eval_data))))

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    train_data = train_data.map(tokenize, batched=True, remove_columns=train_data.column_names)
    eval_data = eval_data.map(tokenize, batched=True, remove_columns=eval_data.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=OUTPUT_PATH,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        report_to="none",
        use_cpu=not torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    print(f"Fine-tuned model saved to: {Path(OUTPUT_PATH).resolve()}")


if __name__ == "__main__":
    main()
