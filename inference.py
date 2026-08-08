"""
inference.py
Load the base model + saved LoRA adapter and run inference from the command line.

Usage:
    python inference.py --instruction "Explain what QLoRA is in simple terms."
"""

import argparse

from unsloth import FastLanguageModel

PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a \
response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def load_model(adapter_path: str, max_seq_length: int = 2048):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,  # local adapter dir or HF hub repo id
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate(model, tokenizer, instruction: str, max_new_tokens: int = 256) -> str:
    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        temperature=0.7,
        top_p=0.9,
    )
    full_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    # Strip the prompt back off so only the completion is returned
    return full_text.split("### Response:")[-1].strip()


def main():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned LoRA adapter.")
    parser.add_argument("--adapter_path", type=str, default="lora_adapters/llama3-8b-custom")
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    model, tokenizer = load_model(args.adapter_path)
    response = generate(model, tokenizer, args.instruction, args.max_new_tokens)

    print("\n--- Response ---")
    print(response)


if __name__ == "__main__":
    main()
