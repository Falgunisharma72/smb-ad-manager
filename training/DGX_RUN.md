# DGX Quickstart — final, compatibility-first

This is the clean run-through for DGX A100 with driver 470. No bitsandbytes, no wandb, no TRL — just `transformers + peft` and a custom minimal GRPO loop.

## The stack (what works on this DGX)

| | Version | Why |
|---|---|---|
| Container | `nvcr.io/nvidia/pytorch:22.12-py3` | Has NVIDIA's torch 1.14.0a0 + CUDA 11.8 forward-compat for driver 470 |
| Python | 3.8 (container default) | Pinned deps all support 3.8 |
| torch | `1.14.0a0+nvidia` (pre-installed) | DO NOT UPGRADE — pip upgrade kills forward-compat |
| transformers | 4.40.2 | Last version supporting Python 3.8 + works with NVIDIA's torch |
| peft | 0.10.0 | LoRA adapters, compatible with transformers 4.40 |
| datasets | 2.14.7 | Compatible with Python 3.8 |
| accelerate | 0.30.1 | Compatible with transformers 4.40 |
| tensorboard | latest | Reward curve logging |

## Steps

### 1. On DGX host, make sure everything is committed + pulled
```bash
cd ~/smb-ad-manager
git pull
```

### 2. Delete old broken container, create fresh one
```bash
docker rm smbad-falguni 2>/dev/null

docker run -it --name smbad-falguni --gpus all --shm-size=16g \
  -v $HOME/smb-ad-manager:/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace \
  nvcr.io/nvidia/pytorch:22.12-py3 \
  bash
```

You should see `root@<hash>:/workspace#`.

### 3. Inside container — install compatible deps
```bash
apt-get update && apt-get install -y git
git config --global --add safe.directory /workspace

pip install -r training/requirements-dgx.txt
```

Watch the output. **It should NOT say "Installing torch..."**. If it does, abort (Ctrl-C) — something picked up a torch-upgrading package.

### 4. Verify torch is STILL NVIDIA's build
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Expected: `1.14.0a0+<something>` and `True`. If it shows `2.1.2+cu118`, stop — the forward-compat is gone.

### 5. Log into HuggingFace
```bash
huggingface-cli login   # paste HF write token
```

(We skip wandb entirely — using TensorBoard.)

### 6. Run SFT warm-start (~5-10 min on A100)
```bash
python training/sft_dgx.py
```

Expected: loss drops over 3 epochs. Final cell prints a sanity JSON generation. Creates `./checkpoints/sft_adapter/`.

### 7. Warm up the env Space in a browser tab
Open https://falgunisharma-smb-ad-manager.hf.space — keep tab open during GRPO.

### 8. Run custom GRPO training (~20-40 min on A100)
```bash
export SPACE_URL="https://falgunisharma-smb-ad-manager.hf.space"
python training/grpo_dgx.py
```

Expected behavior:
- Fetches 50 prompts from the env (~30 sec)
- Prints `[step N] reward_mean=X.XXX reward_best=Y.YYY loss=Z.ZZZZ` every 5 steps
- Reward mean should trend UP over training; starts around 0.3, targets 0.6-0.8
- Saves checkpoint every 50 steps to `./checkpoints/grpo_final/step_XXXX/`
- Final adapter at `./checkpoints/grpo_final/`

### 9. View reward curves via TensorBoard
In a SECOND terminal, ssh into DGX with port forwarding:
```bash
ssh -L 6006:localhost:6006 falguni@103.14.125.72
```

Then inside the container (or on DGX host if you pip-install tensorboard there):
```bash
tensorboard --logdir ./tb_logs --host 0.0.0.0 --port 6006
```

Open `http://localhost:6006` in your laptop's browser. You'll see:
- `reward/mean` — main curve to screenshot
- `reward/best` — per-step max
- `loss/policy` — should be small negative/positive values

### 10. Save results + push to repo
```bash
# Inside container, after training completes
mkdir -p results
# Screenshot the TB charts from your browser → save PNGs under results/ in the mounted /workspace
# E.g., place them in /workspace/results/ which is ~/smb-ad-manager/results on the host
```

Write a `results/training_summary.md`:
```
# Training Run — DGX A100
Date: 2026-04-XX
GPU: 1x A100 80GB (of 8 available)
Runtime: SFT ~8 min, GRPO ~30 min
Initial reward: ~0.XX
Final reward: ~0.YY
Total steps: 200
Model: Qwen 2.5 1.5B Instruct (fp16 + LoRA)
Algorithm: Custom group-relative REINFORCE (GRPO-style)
```

Commit + push:
```bash
cd /workspace
git add results/
git commit -m "results: DGX training run complete"
git push origin main
git push hf main
```

### 11. Push trained adapter to HF Hub (optional but recommended)
```bash
python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', torch_dtype=torch.float16, device_map='cuda')
model = PeftModel.from_pretrained(base, './checkpoints/grpo_final')
model.push_to_hub('Falgunisharma/smb-ad-manager-grpo', private=True)
tok = AutoTokenizer.from_pretrained('./checkpoints/grpo_final')
tok.push_to_hub('Falgunisharma/smb-ad-manager-grpo', private=True)
print('✓ Pushed to https://huggingface.co/Falgunisharma/smb-ad-manager-grpo')
"
```

### 12. Shut down GPU usage to stop billing
Only needed if we were paying. DGX is free, just exit the container:
```bash
exit
```

## If things go wrong

### Reward is flat for 30+ steps
Try in order:
1. Lower learning rate: edit `grpo_dgx.py`, change `LR = 1e-5` to `LR = 5e-6`, re-run
2. Increase temperature: change `TEMPERATURE = 0.7` to `TEMPERATURE = 1.0`
3. Verify env is responding (in Python): `import requests; print(requests.post("https://falgunisharma-smb-ad-manager.hf.space/reset", json={"task_id":"easy"}).status_code)`

### OOM error
Edit `grpo_dgx.py`:
- `BATCH_SIZE = 1`
- `NUM_GENERATIONS = 2` (keep at 2)
- `MAX_NEW_TOKENS = 128` (from 256)

### torch.cuda.is_available() → False after pip install
Rebuild the container — someone installed a torch wheel. Repeat from Step 2.

### Reward function takes forever
HF Space may have cold-started. Open https://falgunisharma-smb-ad-manager.hf.space in a browser and wait for it to wake up (60 sec). Then retry.
