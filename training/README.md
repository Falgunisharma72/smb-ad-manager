# Training pipeline

Two-stage training: **SFT warm-start → GRPO RL**.

## Files

| File | What it does |
|---|---|
| `generate_sft_data.py` | Produces `sft_data.jsonl` — 100 heuristic-generated examples |
| `sft_data.jsonl` | 100 chat-format (system/user/assistant) training rows |
| `sft_warm_start.py` | Stage 1: fine-tune Llama 3.2 1B on the SFT data (Unsloth + LoRA 4-bit) |
| `train_grpo.py` | Stage 2: GRPO RL on top of the SFT adapter, reward comes from HF Space |
| `requirements-training.txt` | Pinned training-time deps |

## Workflow

```bash
# 0. One-time: generate the SFT dataset
python training/generate_sft_data.py     # → training/sft_data.jsonl

# 1. Stage 1 — SFT warm start (on GPU: Colab T4/L4 or DGX A100)
#    Open training/sft_warm_start.py in Colab (it auto-detects # %% cells)
#    OR run cell-by-cell in VS Code's Python interactive mode.
#    Output: ./checkpoints/sft_adapter/

# 2. Stage 2 — GRPO training
#    Prerequisites: HF Space must be deployed + public.
#    Open training/train_grpo.py
#    Set SPACE_URL env var. Run all cells.
#    Output: ./checkpoints/grpo_final/
```

## Citations

- **GRPO:** DeepSeek-R1, arxiv [2501.12948](https://arxiv.org/abs/2501.12948)
- **2-GRPO:** "It Takes Two: Your GRPO Is Secretly DPO", arxiv [2510.00977](https://arxiv.org/abs/2510.00977)
- **Unsloth:** [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)
- **TRL:** [huggingface.co/docs/trl](https://huggingface.co/docs/trl)

## Model choice

Default: **Qwen 2.5 1.5B Instruct** (`Qwen/Qwen2.5-1.5B-Instruct`)
- No license gate (instant download)
- Strong on structured JSON output
- ~1.5B params — fits on any modern GPU with 4-bit quantization

Alternative models you can swap in (edit `MODEL_NAME` / `BASE_MODEL` in the scripts):
- `meta-llama/Llama-3.2-1B-Instruct` (gated — requires accepting license at HF)
- `Qwen/Qwen2.5-3B-Instruct` (more capable, ~2× slower to train)

## Runtime expectations

| Stage | Colab T4 (free) | Colab L4 (Pro) | DGX A100 (Unsloth) | DGX A100 (non-Unsloth) |
|---|---|---|---|---|
| SFT warm-start (3 epochs × 100 examples) | ~20 min | ~8 min | ~3 min | ~5-10 min |
| GRPO (200 steps, group size 2) | ~4-5 hrs | ~2 hrs | ~30-45 min | ~25-40 min |

## Logs

- W&B project: `smb-ad-manager`
- Reward columns logged separately (5 of them) so judges can see each
  component's curve — and catch reward hacking visually.
