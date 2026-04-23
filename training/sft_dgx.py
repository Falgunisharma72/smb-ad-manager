"""SFT warm-start — DGX-compatible variant.

Works inside nvcr.io/nvidia/pytorch:22.12-py3 on driver 470 / Python 3.8.
No bitsandbytes (fp16 full precision LoRA — fits on A100 80GB easily).
No TRL (uses transformers.Trainer directly).
No wandb (uses tensorboard).

Expected runtime on A100: ~5-10 min for 3 epochs × 100 examples.
Output: ./checkpoints/sft_adapter/
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ─── Config ──────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SFT_DATA_PATH = "training/sft_data.jsonl"
OUTPUT_DIR = "./checkpoints/sft_adapter"
TB_LOG_DIR = "./tb_logs/sft"
EPOCHS = 3
LEARNING_RATE = 2e-4
PER_DEVICE_BATCH_SIZE = 2
GRAD_ACCUMULATION = 4
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 16

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(TB_LOG_DIR).mkdir(parents=True, exist_ok=True)

# ─── Load model + tokenizer ──────────────────────────────────────────────
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

print(f"Config: model={MODEL_NAME}, epochs={EPOCHS}, eff_batch={PER_DEVICE_BATCH_SIZE * GRAD_ACCUMULATION}")
print(f"CUDA: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map={"": 0},       # pin to GPU 0 (of 8 A100s)
)

# LoRA adapter
peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Gradient checkpointing + input-grad so LoRA works
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# ─── Load + tokenize SFT data ────────────────────────────────────────────
raw = load_dataset("json", data_files=SFT_DATA_PATH, split="train")
print(f"Loaded {len(raw)} SFT examples")


def tokenize_example(ex):
    # Apply the model's chat template
    text = tokenizer.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False,
    )
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_attention_mask=True,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


dataset = raw.map(tokenize_example, remove_columns=raw.column_names)
print(f"Tokenized. Sample length: {len(dataset[0]['input_ids'])} tokens")

# ─── Train ───────────────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    logging_steps=5,
    save_strategy="epoch",
    fp16=True,
    gradient_checkpointing=True,
    report_to=["tensorboard"],
    logging_dir=TB_LOG_DIR,
    save_safetensors=True,
    seed=42,
    remove_unused_columns=False,
)

collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, padding=True, return_tensors="pt",
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    train_dataset=dataset,
    data_collator=collator,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n✓ Saved SFT adapter to {OUTPUT_DIR}")

# ─── Sanity-check generation ─────────────────────────────────────────────
print("\n=== Sanity-check generation ===")
model.eval()
test_messages = [
    {"role": "system", "content": "You are an AI Ad Manager. Emit a JSON action."},
    {"role": "user", "content": (
        "Business: Priya's Candles. Budget remaining: ₹8500.\n"
        "Active campaigns:\n  - c001: daily ₹400, active\n"
        "Latest metrics:\n  - c001: imp=3500, clicks=42, conv=5, spend=₹400, rev=₹2200, ROAS=5.5x\n"
        "What action?"
    )},
]
prompt = tokenizer.apply_chat_template(
    test_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
).to("cuda")
with torch.no_grad():
    out = model.generate(
        prompt,
        max_new_tokens=200,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
resp = tokenizer.decode(out[0, prompt.shape[-1]:], skip_special_tokens=True)
print(resp)
print("\n=== Done ===")
