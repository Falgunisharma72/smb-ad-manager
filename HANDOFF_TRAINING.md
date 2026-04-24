# Training Handoff — Pick Your Path

**Project:** SMB Ad Manager RL Environment · Scaler OpenEnv Hackathon · April 2026

This document gives you **two complete paths** to train the model. Pick based on your situation. Read all of **Part 1** (common prerequisites), then skip to **Part 2 (DGX)** OR **Part 3 (HF Compute)**.

---

## Which path should YOU take?

| If you have… | Use this path | Time | Cost |
|---|---|---|---|
| DGX A100 SSH access (IP + creds) | **Path A — DGX A100** | ~2 hrs setup + training | Free |
| Only a laptop, credit card for compute | **Path B — HF GPU Space** | ~20 min setup + 2-3 hrs training | ~$3-5 |

**If you have BOTH:** still prefer Path B (HF Compute). DGX has driver limitations that cost us time — HF "just works."

---

## Part 1 — Prerequisites (common to both paths, 15 min)

### 1.1 — Create HuggingFace account
- https://huggingface.co/join
- Verify email

### 1.2 — Create HF Write-scope token
- https://huggingface.co/settings/tokens
- Click **"Create new token"** → Type: **Write** (NOT fine-grained) → name: `hackathon-train`
- Copy token (starts with `hf_...`). **Save in password manager. Never paste in chat.**

### 1.3 — Add HF billing
- https://huggingface.co/billing → Add payment card (only needed if using Path B).
- Path A users: skip this.

### 1.4 — Clone the project repo
```bash
git clone https://github.com/Falgunisharma72/smb-ad-manager.git
cd smb-ad-manager
```

### 1.5 — Verify the training files are there
```bash
ls training/
```
Should show:
- `sft_dgx.py`
- `grpo_dgx.py`
- `requirements-dgx.txt`
- `sft_data.jsonl` (100 examples)
- `sft_warm_start_nounsloth.py` (HF compute fallback)
- `train_grpo_nounsloth.py` (HF compute fallback)
- `DGX_RUN.md`
- `HANDOFF_TRAINING.md` (this document)

---

# Part 2 — Path A: DGX A100

## DGX is free and fast **IF** you can get past the driver limitations. We've already figured those out — follow this carefully.

### A.1 — Connect to DGX via VS Code (recommended)

1. Install VS Code on Windows/Mac + the **"Remote - SSH"** extension
2. Open SSH config file: `Cmd/Ctrl+Shift+P` → "Remote-SSH: Open SSH Configuration File" → `~/.ssh/config`
3. Add this block (replace `YOUR_USERNAME`):
   ```
   Host dgx-a100
       HostName 103.14.125.72
       User YOUR_USERNAME
       Port 22
   ```
4. `Cmd/Ctrl+Shift+P` → "Remote-SSH: Connect to Host" → select `dgx-a100` → enter password
5. You should see `SSH: dgx-a100` in the bottom-left of VS Code

### A.2 — Verify you have an A100
Open terminal in VS Code (`Ctrl+` backtick), then:
```bash
nvidia-smi
```
You should see 8 × NVIDIA A100-SXM4 80GB and driver version `470.xxx`.

### A.3 — Critical pre-work: sideload the model (10-20 min)

**DGX's network blocks HuggingFace's large-file CDN.** Config files work; 3 GB model weights get stuck at 0%. So we download on a fast machine and copy.

**On your laptop (not DGX):**
```bash
pip install huggingface_hub
export HF_TOKEN=your_write_token

# HF 1.x uses `hf`, older uses `huggingface-cli`. Use whichever works:
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir ~/qwen25-model --token $HF_TOKEN
# OR:
python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='Qwen/Qwen2.5-1.5B-Instruct',
    local_dir='/home/<you>/qwen25-model',
    token=os.environ['HF_TOKEN'],
)
"
```

Verify: `du -sh ~/qwen25-model/` → ~2.9 GB.

Upload to DGX:
```bash
scp -r ~/qwen25-model YOUR_USERNAME@103.14.125.72:~/smb-ad-manager/
```

### A.4 — On DGX: clone the repo if not there
```bash
cd ~
ls smb-ad-manager 2>/dev/null || git clone https://github.com/Falgunisharma72/smb-ad-manager.git
cd smb-ad-manager
git pull
```

### A.5 — Pull the NGC PyTorch container (first time only)
```bash
docker pull nvcr.io/nvidia/pytorch:22.12-py3
```
~10 min download (~20 GB). This is the **only container** that works on driver 470 with modern-ish PyTorch (1.14.0a0 + CUDA 11.8 forward-compat).

### A.6 — Create your container
Use your own name to keep it isolated from other users:
```bash
docker run -it --name smbad-YOURNAME --gpus all --shm-size=16g \
  -v $HOME/smb-ad-manager:/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace \
  nvcr.io/nvidia/pytorch:22.12-py3 \
  bash
```
Prompt changes to `root@<hash>:/workspace#`.

### A.7 — Fix volume permissions (CRITICAL — skip and git pull breaks later)
```bash
apt-get update && apt-get install -y git
git config --global --add safe.directory /workspace
```

### A.8 — Verify NVIDIA's torch is alive (with CUDA forward-compat)
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
**Expected:** `1.14.0a0+410ce96 True`

If it says `2.x` or `False` — something upgraded torch, the compat is dead. Exit, `docker rm smbad-YOURNAME`, rerun A.6.

### A.9 — Install deps (CAREFUL — do NOT upgrade torch!)
```bash
pip install -r training/requirements-dgx.txt
```
**Watch the output.** Look for `Requirement already satisfied: torch...` — means pip kept NVIDIA's torch. If you ever see `Installing: torch-2.x`, abort with Ctrl+C.

### A.10 — Verify deps installed without breaking torch
```bash
python -c "import torch, transformers, peft; print('torch:', torch.__version__, torch.cuda.is_available()); print('transformers:', transformers.__version__); print('peft:', peft.__version__)"
```
Should show: `torch: 1.14.0a0+... True`, `transformers: 4.40.2`, `peft: 0.10.0`.

### A.11 — Log into HuggingFace inside container
```bash
huggingface-cli login
```
Paste your NEW (rotated) HF token. `Y` for git credential.

### A.12 — Point training scripts at the sideloaded model
```bash
sed -i 's|MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"|MODEL_NAME = "/workspace/qwen25-model"|' training/sft_dgx.py
sed -i 's|BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"|BASE_MODEL = "/workspace/qwen25-model"|' training/grpo_dgx.py
grep -E "^(MODEL_NAME|BASE_MODEL)" training/sft_dgx.py training/grpo_dgx.py
```
Should print `/workspace/qwen25-model` for both.

### A.13 — Run SFT warm-start (~5-10 min)
```bash
python training/sft_dgx.py
```
Expected: loss drops over 3 epochs. Final line: `✓ Saved SFT adapter to ./checkpoints/sft_adapter`.

### A.14 — Warm up env, then run GRPO (~20-40 min)
Open https://falgunisharma-smb-ad-manager.hf.space in a browser tab (any browser, any machine). Leave open.

Then:
```bash
export SPACE_URL="https://falgunisharma-smb-ad-manager.hf.space"
python training/grpo_dgx.py
```
Watch console for `[step N] reward_mean=X.XXX` — should trend upward.

### A.15 — Skip to **Part 4 — save results**

### DGX troubleshooting

| Symptom | Fix |
|---|---|
| `docker exec` says "container not running" | `docker start -ai smbad-YOURNAME` |
| `git pull` says permission denied | On host: `docker run --rm -v $HOME/smb-ad-manager:/workspace nvcr.io/nvidia/pytorch:22.12-py3 chown -R $(id -u):$(id -g) /workspace` |
| Model download stuck at 0% | This is the known issue — you MUST sideload (A.3) |
| `sed` says "no input files" | You split the command across lines — put it all on one line |
| After `pip install ...`, CUDA is False | Stock torch overwrote NVIDIA's — recreate the container (A.6) |
| `wandb login` rejects key as "86 chars" | Known — use TensorBoard, already set in our scripts |
| Training reward is flat for 30+ steps | In `grpo_dgx.py`, lower `LR = 1e-5` → `5e-6`, re-run |

---

# Part 3 — Path B: HuggingFace GPU Space ($3-5, ~20 min setup)

## B.1 — Create an HF GPU Jupyter Space

1. Go to https://huggingface.co/new-space
2. Owner: your HF username
3. Name: `smb-training-yourname`
4. SDK: **Jupyter Notebook**
5. Hardware: **Nvidia L4 small** ($0.80/hr, ~24 GB VRAM)
6. Visibility: **Private**
7. Click **Create Space**
8. Wait ~90 sec for it to boot, then click **Open in JupyterLab**

## B.2 — Open a terminal in the Space
File → New → Terminal

## B.3 — Clone the repo
```bash
cd ~
git clone https://github.com/Falgunisharma72/smb-ad-manager.git
cd smb-ad-manager
ls training/
```

## B.4 — Install training dependencies (full stack, including Unsloth)
```bash
pip install -r training/requirements-training-nounsloth.txt
```
Unlike DGX, L4 has modern driver — you can use the full stack.

## B.5 — Log into HuggingFace + W&B
```bash
huggingface-cli login    # paste your HF token
wandb login              # paste your W&B API key (from https://wandb.ai/authorize)
```

## B.6 — Run SFT (~20 min on L4)
```bash
python training/sft_warm_start_nounsloth.py
```
**No sideload needed** — HF GPU has direct access to model weights, download is fast.

Expected: loss drops. Final line `✓ Saved SFT adapter to ./checkpoints/sft_adapter`.

## B.7 — Warm up env, then run GRPO (~2 hrs on L4)
```bash
export SPACE_URL="https://falgunisharma-smb-ad-manager.hf.space"
python training/train_grpo_nounsloth.py
```
Watch W&B dashboard (link prints at start) for live reward curves.

## B.8 — Skip to **Part 4 — save results**

### HF Compute troubleshooting

| Symptom | Fix |
|---|---|
| Space won't start | Check hardware tier — use L4 small, not CPU |
| OOM on L4 | Edit config cell: `PER_DEVICE_BATCH_SIZE = 1` |
| Training speed slow | L4 is ~2× slower than A100 — normal |
| **STOP** | After training ends, go to Space Settings → **Pause Space** to stop billing |

---

# Part 4 — Save results + push back to repo (common to both paths, 20 min)

### 4.1 — Screenshot reward curves
- **DGX (TensorBoard):** In a new terminal on your laptop:
  ```bash
  ssh -L 6006:localhost:6006 YOUR_USERNAME@103.14.125.72
  # Then in container or host:
  tensorboard --logdir ./tb_logs --host 0.0.0.0 --port 6006
  ```
  Open `http://localhost:6006` in browser → screenshot charts.

- **HF Compute (W&B):** Go to your W&B project → screenshot these charts:
  - `reward/mean` (main curve)
  - `reward/r1_roas_improvement`, `reward/r3_format_compliance`, `reward/r5_no_cheating`
  - Save as PNGs

### 4.2 — Save screenshots in the repo
```bash
cd /workspace  # or wherever you cloned the repo
mkdir -p results
# Drop your 4 PNGs into results/ — named:
#   reward_curve_total.png
#   reward_curve_r1.png
#   reward_curve_r3.png
#   reward_curve_r5.png
```

### 4.3 — Write a training summary
Create `results/training_summary.md`:
```markdown
# Training Run Summary

**Date:** 2026-04-XX
**Run by:** YourName
**Compute:** [DGX A100 / HF L4]
**Total cost:** $X.XX (0 for DGX)
**Total wall-clock:** ~Y hours

## SFT warm-start
- Model: Qwen 2.5 1.5B Instruct
- Examples: 100
- Epochs: 3
- Final loss: <value from logs>

## GRPO training
- Algorithm: Custom group-relative REINFORCE (DGX) / 2-GRPO TRL (HF)
- Steps: 200
- Starting reward (step 0): 0.XX
- Final reward (step 200): 0.XX
- Best reward: 0.XX

## Dashboard URL (optional)
https://wandb.ai/.../runs/...

## Trained model on HF Hub
https://huggingface.co/YOUR-USERNAME/smb-ad-manager-grpo
```

### 4.4 — Push trained model to HuggingFace Hub
Inside your notebook/terminal:
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Adapt paths to your setup (DGX uses /workspace, HF Compute uses ~/smb-ad-manager)
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",   # or "/workspace/qwen25-model" on DGX
    torch_dtype=torch.float16, device_map="cuda",
)
model = PeftModel.from_pretrained(base, "./checkpoints/grpo_final")
model.push_to_hub("YOUR-HF-USERNAME/smb-ad-manager-grpo", private=True)
tokenizer = AutoTokenizer.from_pretrained("./checkpoints/grpo_final")
tokenizer.push_to_hub("YOUR-HF-USERNAME/smb-ad-manager-grpo", private=True)
```

### 4.5 — Commit + push results
```bash
cd /workspace  # or ~/smb-ad-manager
git add results/
git commit -m "results: training run complete — $(date +%F) by YourName"

# Push to GitHub:
git push origin main

# If you have HF Space remote configured:
git push hf main 2>/dev/null || echo "no HF remote, skipping"
```

### 4.6 — Shut down compute (CRITICAL for HF path)
- **DGX:** just `exit` the container. No billing.
- **HF L4:** **Pause the Space** at `https://huggingface.co/spaces/<YOU>/smb-training-<name>/settings` OR change hardware back to **CPU basic** (free). **If you forget this, you'll be billed until you return.**

---

# Part 5 — Final checklist

Before walking away, confirm ALL of these are true:

- [ ] 4 reward curve PNGs are in `results/` and committed to the GitHub repo
- [ ] `results/training_summary.md` has real numbers filled in
- [ ] GRPO adapter pushed to HF Hub (link added to summary)
- [ ] SFT adapter saved either locally or pushed to HF Hub
- [ ] (HF path only) GPU Space is PAUSED or DELETED — billing stopped
- [ ] Backup: download `./checkpoints/grpo_final/` as zip to your laptop just in case

---

# Part 6 — What "done" looks like

After your run is complete and pushed:
1. https://github.com/Falgunisharma72/smb-ad-manager/tree/main/results has 5 files (4 PNGs + summary.md)
2. https://huggingface.co/YOUR-USERNAME/smb-ad-manager-grpo exists and is viewable
3. `results/training_summary.md` shows reward climbing from ~0.3 to ~0.7+
4. You have a zipped backup of the checkpoint on your laptop

---

# Quick-reference: key URLs

- GitHub: https://github.com/Falgunisharma72/smb-ad-manager
- Live env (must be warm before GRPO): https://falgunisharma-smb-ad-manager.hf.space
- DGX SSH: `103.14.125.72` port 22
- Your W&B project (HF path): https://wandb.ai/YOUR-USERNAME/smb-ad-manager

---

# Security reminder (applies to both paths)

- **NEVER paste tokens (HF, OpenAI, W&B) into chat, Slack, email, or screenshots.**
- Store tokens in a password manager.
- If you think a token leaked: immediately revoke at `https://huggingface.co/settings/tokens`, regenerate, re-login everywhere.

---

*This document lives at `TRAINING_HANDOFF.md` in the repo root. Keep it updated as you learn new gotchas.*
