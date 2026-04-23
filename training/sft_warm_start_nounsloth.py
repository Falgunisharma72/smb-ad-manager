# %% [markdown]
# # SFT Warm-Start (non-Unsloth variant) — for CUDA 11.x / driver 470
#
# Same contract as `sft_warm_start.py` but uses only transformers + peft + bitsandbytes.
# Slower than Unsloth by ~2× but totally sufficient on A100.
#
# **Use this on DGX (driver 470 / CUDA 11.4) where Unsloth can't run.**
#
# Expected runtime on A100 80GB: ~5-10 min for 3 epochs × 100 examples.
#
# Output: `./checkpoints/sft_adapter/` (LoRA adapter weights, ~10MB)
#
# This file is jupytext `# %%` format. Run as:
#   python training/sft_warm_start_nounsloth.py
# Or open cell-by-cell in VS Code / JupyterLab.

# %%
# Cell 1 — Config
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"   # no license gate, same chat-template interface
MAX_SEQ_LENGTH = 2048
SFT_DATA_PATH = "training/sft_data.jsonl"          # relative to repo root
OUTPUT_DIR = "./checkpoints/sft_adapter"
EPOCHS = 3
LEARNING_RATE = 2e-4
PER_DEVICE_BATCH_SIZE = 2
GRAD_ACCUMULATION = 4
LORA_R = 16
LORA_ALPHA = 16

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
print(f"Config: model={MODEL_NAME}, epochs={EPOCHS}, eff_batch={PER_DEVICE_BATCH_SIZE * GRAD_ACCUMULATION}")

# %%
# Cell 2 — Load model in 4-bit (bitsandbytes) and attach LoRA (peft)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},           # pin to GPU 0
    torch_dtype=torch.float16,
)
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# %%
# Cell 3 — Load + format SFT data (chat template)
from datasets import load_dataset

dataset = load_dataset("json", data_files=SFT_DATA_PATH, split="train")
print(f"Loaded {len(dataset)} examples")

def format_example(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = dataset.map(format_example)
print("\nFormatted sample (first 400 chars):")
print(dataset[0]["text"][:400])

# %%
# Cell 4 — Train with TRL SFTTrainer
from trl import SFTTrainer, SFTConfig

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_steps=5,
    save_strategy="epoch",
    report_to=["tensorboard"],    # use tensorboard; swap to ["wandb"] if you have working wandb
    logging_dir="./tb_logs/sft",
    dataset_text_field="text",
    max_length=MAX_SEQ_LENGTH,
    packing=False,
    fp16=True,                   # A100 also supports bf16; fp16 is broadest compat
    seed=42,
    save_safetensors=True,
    gradient_checkpointing=True,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=sft_config,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n✓ Saved SFT adapter to {OUTPUT_DIR}")

# %%
# Cell 5 — Sanity-check generation
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
inputs = tokenizer.apply_chat_template(
    test_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
).to("cuda")
with torch.no_grad():
    out = model.generate(
        inputs, max_new_tokens=200, temperature=0.3, do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
resp = tokenizer.decode(out[0, inputs.shape[-1]:], skip_special_tokens=True)
print("\n=== Sanity-check generation ===")
print(resp)

# %%
# Cell 6 — Optional: push adapter to HF Hub (uncomment to use)
# model.push_to_hub("YOUR-HF-USERNAME/smb-ad-manager-sft", private=True)
# tokenizer.push_to_hub("YOUR-HF-USERNAME/smb-ad-manager-sft", private=True)
