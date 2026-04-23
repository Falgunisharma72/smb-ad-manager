# DGX Docker Quickstart — non-Unsloth variant

**Use this path when running on DGX with driver 470 / CUDA 11.4.** Unsloth isn't an
option there (needs driver 525+), but we have 8× A100 80GB which is more than
enough power without it.

## 1. Pull the image (once per DGX host)

```bash
docker pull pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel
```

## 2. Run the container interactively with GPU + repo + HF cache mounted

```bash
docker run --gpus all -it --rm \
  --shm-size=16g \
  -v $HOME/smb-ad-manager:/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/.netrc:/root/.netrc \
  -w /workspace \
  pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel \
  bash
```

- `--gpus all` exposes all 8 A100s to the container
- `--shm-size=16g` avoids "shared memory" errors in DataLoader / generation
- `-v ... .netrc` carries your W&B credentials if you've done `wandb login` on the host

## 3. Inside the container — install training deps

```bash
pip install -r training/requirements-training-nounsloth.txt
```

## 4. Log into HuggingFace + W&B inside the container

```bash
huggingface-cli login           # paste HF write token
wandb login                     # paste W&B API key
```
(Skip `wandb login` if `.netrc` was mounted from host and already has the key.)

## 5. Verify GPU is visible

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: 2.1.2+cu118 True NVIDIA A100-SXM4-80GB
```

## 6. Run SFT warm-start (~5-10 min on A100)

```bash
python training/sft_warm_start_nounsloth.py
```

**Expected:** Loss drops across 3 epochs. Final cell prints a sanity-check JSON
generation. Creates `checkpoints/sft_adapter/`.

## 7. Warm up the env Space, then run GRPO (~20-40 min on A100)

In your browser, open https://falgunisharma-smb-ad-manager.hf.space once (60s cold start).

Then in the container:
```bash
export SPACE_URL="https://falgunisharma-smb-ad-manager.hf.space"
python training/train_grpo_nounsloth.py
```

**Expected:** W&B link prints early. Reward climbs from ~0.3 to ~0.7-0.9. Creates
`checkpoints/grpo_final/`.

## 8. If the reward curve goes flat — use the Training Runbook's Fix A-E

See `TRAINING_RUNBOOK.md` Part 5-4.

## 9. Save + push (same as Runbook Part 6)

Screenshots → `results/` folder → commit → push to both GitHub and HF Space.

## Notes on this variant vs. the Unsloth variant

| | Unsloth variant | non-Unsloth (this) |
|---|---|---|
| GPU requirement | CUDA 12.x (driver 525+) | CUDA 11.8 (driver 450+) ✓ DGX 470 works |
| Speed vs. base | ~2× faster | baseline |
| Training time on A100 | ~15-20 min | ~25-40 min |
| Library stack | `unsloth` + `trl` + `peft` | `transformers` + `trl` + `peft` + `bitsandbytes` |
| Final model quality | Same | Same |

The 10-15 min extra training time is fine — we still finish in under an hour total.
