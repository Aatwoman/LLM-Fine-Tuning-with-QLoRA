# 🦙 LLM Fine-Tuning with QLoRA (Llama 3 8B + Unsloth)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Unsloth](https://img.shields.io/badge/Unsloth-2x_faster_finetuning-brightgreen)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers%20%7C%20PEFT%20%7C%20TRL-yellow)
![Colab](https://img.shields.io/badge/Google_Colab-Free_T4-F9AB00?logo=googlecolab)
![License](https://img.shields.io/badge/License-MIT-green)

Fine-tunes `Llama 3 8B` on a custom instruction dataset using 4-bit QLoRA via Unsloth — runs end-to-end on a **free Colab T4 GPU**.

## Why QLoRA + Unsloth

Full fine-tuning an 8B model needs ~60GB+ VRAM. QLoRA loads the base model in 4-bit precision and trains only small low-rank adapter matrices (a few % of total parameters), cutting memory needs to what a single T4 (16GB) can handle. Unsloth patches the attention/MLP kernels for ~2x faster training and ~60% less VRAM versus vanilla PEFT.

## Project structure

```
llm-lora-finetuning/
├── finetune_llama3_qlora.ipynb   # Main Colab training notebook
├── data_prep.py                   # Converts raw JSONL into Alpaca-format training text
├── inference.py                   # CLI inference with the saved adapter
├── data/
│   └── sample.jsonl                # 3 example records — replace with your dataset
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

### Option A — Google Colab (recommended, free GPU)
1. Upload `finetune_llama3_qlora.ipynb` to [Colab](https://colab.research.google.com/)
2. Runtime → Change runtime type → **T4 GPU**
3. Upload your `data/processed/train.jsonl` and `val.jsonl` (generate locally with `data_prep.py`, or point `data_prep.py`'s output at a Colab-mounted Drive)
4. Run all cells

### Option B — Local / cloud GPU
```bash
git clone https://github.com/<your-username>/llm-lora-finetuning.git
cd llm-lora-finetuning
pip install -r requirements.txt

python data_prep.py --raw_path data/sample.jsonl --out_dir data/processed
jupyter nbconvert --to notebook --execute finetune_llama3_qlora.ipynb
python inference.py --instruction "Explain what QLoRA is in simple terms."
```

## Dataset format

Raw input is Alpaca-style JSONL — swap in your own domain data (medical Q&A, code comments, support tickets, etc.):

```json
{"instruction": "...", "input": "", "output": "..."}
```

`data_prep.py` formats each record into the prompt template the model is trained on and does a 95/5 train/val split.

## Training config (defaults)

| Param | Value |
|---|---|
| Base model | `unsloth/llama-3-8b-bnb-4bit` |
| LoRA rank (r) | 16 |
| LoRA alpha | 16 |
| Target modules | q/k/v/o_proj, gate/up/down_proj |
| Batch size | 2 (× 4 grad accumulation) |
| Epochs | 3 |
| Learning rate | 2e-4 |
| Optimizer | adamw_8bit |

## Tech stack

`HuggingFace Transformers` · `PEFT` · `Unsloth` · `TRL (SFTTrainer)` · `bitsandbytes` · `Google Colab`

## Possible extensions

- Add DPO/ORPO alignment stage on top of the SFT adapter
- Evaluate with a held-out benchmark (e.g. domain-specific accuracy, perplexity)
- Merge adapter into the base model (`model.merge_and_unload()`) and quantize to GGUF for `llama.cpp` deployment
- Swap in Mistral 7B or a smaller model for faster iteration

---

### Resume bullet points

- Fine-tuned Llama 3 8B on a custom instruction dataset using QLoRA (4-bit quantization + low-rank adapters), reducing GPU memory requirements enough to train on a free Colab T4
- Built a reusable data pipeline converting raw instruction/output pairs into model-ready training format, with automated train/validation splitting
- Achieved ~2x training speedup and ~60% lower VRAM usage versus standard PEFT by using Unsloth's optimized kernels

### Recruiter talking points

- **What it demonstrates:** practical, resource-constrained fine-tuning — the kind of problem-solving needed when you don't have a multi-GPU cluster.
- **Design decisions worth discussing:** why QLoRA over full fine-tuning or standard LoRA; rank/alpha trade-offs; why particular target modules were chosen.
- **What you'd improve at scale:** multi-GPU training with DeepSpeed/FSDP, larger curated datasets, automated eval harness, adapter merging + quantized deployment.
