# %% [markdown]
# # SFT Warm-Start for SMB Ad Manager Agent
#
# Pre-trains a small Llama (3.2 1B Instruct, Unsloth-optimized) on our 100
# heuristic-generated examples. Output: a LoRA adapter we'll initialize GRPO
# from.
#
# **Why SFT before RL?** Small instruct models emit malformed JSON under RL
# pressure. SFT gives the policy a stable format prior so GRPO focuses on
# *behavior* improvements, not format compliance.
#
# **Expected runtime:**
# - Colab T4 (free): ~20 min
# - L4 (Pro): ~8 min
# - DGX A100: ~3 min
#
# **Output:** `./checkpoints/sft_adapter/` (LoRA weights, ~10MB)
#
# Open this file in Colab via File → Upload notebook, OR run in VS Code as an
# interactive script (`# %%` cells are auto-detected).

# %%
# Cell 1 — Install deps (Colab)
# Uncomment these if running in Colab. Skip if running on DGX with deps already installed.
# !pip install -qqq unsloth trl peft bitsandbytes wandb datasets

# %%
# Cell 2 — Config
from pathlib import Path

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"   # no license gate, swap to -7B on A100
MAX_SEQ_LENGTH = 2048
SFT_DATA_PATH = "sft_data.jsonl"                # relative to the notebook
OUTPUT_DIR = "./checkpoints/sft_adapter"
EPOCHS = 3
LEARNING_RATE = 2e-4
PER_DEVICE_BATCH_SIZE = 2
GRAD_ACCUMULATION = 4      # effective batch = 8
LOAD_IN_4BIT = True

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
print(f"Config: model={MODEL_NAME}, epochs={EPOCHS}, batch={PER_DEVICE_BATCH_SIZE * GRAD_ACCUMULATION}")

# %%
# Cell 3 — Load model with Unsloth (4-bit LoRA)
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
)

# Attach LoRA adapters. These are what we train + save.
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# %%
# Cell 4 — Load SFT data
from datasets import load_dataset

dataset = load_dataset("json", data_files=SFT_DATA_PATH, split="train")
print(f"Loaded {len(dataset)} examples")
print(f"First example: {dataset[0]['messages'][0]['content'][:200]}...")

# %%
# Cell 5 — Format with Llama 3 chat template
def format_example(example):
    # messages is a list of {role, content}. Apply the model's chat template.
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = dataset.map(format_example)
print("\nFormatted sample:")
print(dataset[0]["text"][:500])

# %%
# Cell 6 — Train with TRL's SFTTrainer
from trl import SFTTrainer, SFTConfig

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_steps=5,
    save_strategy="epoch",
    report_to=["wandb"],    # remove if wandb isn't set up
    dataset_text_field="text",
    max_length=MAX_SEQ_LENGTH,
    packing=False,
    bf16=True,
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=sft_config,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"\n✓ Saved SFT adapter to {OUTPUT_DIR}")

# %%
# Cell 7 — Quick sanity check: run a generation on a held-out prompt
FastLanguageModel.for_inference(model)

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

out = model.generate(inputs, max_new_tokens=200, temperature=0.3, do_sample=True)
resp = tokenizer.decode(out[0, inputs.shape[-1]:], skip_special_tokens=True)
print("\n=== Sanity-check generation ===")
print(resp)

# %%
# Cell 8 — Push adapter to HF Hub (optional, for GRPO to pick it up)
# Uncomment if you want to push to HuggingFace.
#
# model.push_to_hub("Falgunisharma/smb-ad-manager-sft", private=True)
# tokenizer.push_to_hub("Falgunisharma/smb-ad-manager-sft", private=True)
