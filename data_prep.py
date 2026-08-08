"""
data_prep.py
Prepares a custom instruction dataset for QLoRA fine-tuning.

Expects a raw JSONL file where each line has {"instruction", "input", "output"}
(the classic Alpaca-style schema) and converts it into the chat-formatted
text string Llama 3's tokenizer expects, ready for TRL's SFTTrainer.

Example raw record:
{"instruction": "Explain the symptoms of type 2 diabetes.",
 "input": "",
 "output": "Common symptoms include increased thirst, frequent urination..."}
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, load_dataset

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input \
that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

ALPACA_PROMPT_NO_INPUT = """Below is an instruction that describes a task. Write a \
response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""


def format_example(example: dict, eos_token: str) -> dict:
    """Render one record into the final training string."""
    if example.get("input", "").strip():
        text = ALPACA_PROMPT.format(
            instruction=example["instruction"],
            input=example["input"],
            output=example["output"],
        )
    else:
        text = ALPACA_PROMPT_NO_INPUT.format(
            instruction=example["instruction"],
            output=example["output"],
        )
    return {"text": text + eos_token}


def load_and_format(raw_path: str, eos_token: str = "<|end_of_text|>") -> Dataset:
    """Load a raw JSONL file and return a HF Dataset with a single 'text' column."""
    dataset = load_dataset("json", data_files=raw_path, split="train")
    dataset = dataset.map(lambda ex: format_example(ex, eos_token))
    return dataset.remove_columns(
        [c for c in dataset.column_names if c not in ("text",)]
    )


def train_val_split(dataset: Dataset, val_fraction: float = 0.05, seed: int = 42):
    split = dataset.train_test_split(test_size=val_fraction, seed=seed)
    return split["train"], split["test"]


def main():
    parser = argparse.ArgumentParser(description="Prepare instruction data for QLoRA fine-tuning.")
    parser.add_argument("--raw_path", type=str, default="data/sample.jsonl")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--val_fraction", type=float, default=0.05)
    args = parser.parse_args()

    dataset = load_and_format(args.raw_path)
    train_ds, val_ds = train_val_split(dataset, args.val_fraction)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ds.to_json(out_dir / "train.jsonl")
    val_ds.to_json(out_dir / "val.jsonl")

    print(f"Train examples: {len(train_ds)} | Val examples: {len(val_ds)}")
    print(f"Written to {out_dir}/train.jsonl and {out_dir}/val.jsonl")


if __name__ == "__main__":
    main()
