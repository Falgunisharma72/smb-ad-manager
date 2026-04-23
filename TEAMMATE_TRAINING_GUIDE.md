# Training Guide — For Sarthak / Shrishty

**Project:** SMB Ad Manager · Scaler OpenEnv Hackathon
**Your job:** Run the training on a cloud GPU, save the results, tell Falguni when done.
**Estimated total time:** 2–3 hours (most is waiting for training to run).

---

## What Falguni has already built (so you know what you're running)

- A full RL training environment live at https://falgunisharma-smb-ad-manager.hf.space
- A small Llama model agent (Llama 3.2 1B) that can make Meta Ads decisions
- 100 pre-made training examples
- Two ready-to-run notebooks:
  1. `sft_warm_start.py` — teaches the model the right output format (~30–45 min)
  2. `train_grpo.py` — reinforcement learning on the live environment (~45 min – 2 hrs)

**Your job** = run these two notebooks, save the output, screenshot the graphs.

---

## Step 1 — Get the code on your machine (5 min)

```bash
git clone https://github.com/Falgunisharma72/smb-ad-manager.git
cd smb-ad-manager
```

Verify files are there:
```bash
ls training/
```
You should see: `generate_sft_data.py`, `sft_data.jsonl`, `sft_warm_start.py`, `train_grpo.py`, `README.md`.

---

## Step 2 — Make accounts (if you don't already have them)

1. **HuggingFace account** — https://huggingface.co/join (free)
   - Get your access token: https://huggingface.co/settings/tokens (make it "Write" type)
2. **Weights & Biases account** — https://wandb.ai/signup (free)
   - Copy your API key: https://wandb.ai/authorize
3. **Colab Pro** ($10/month) OR **HuggingFace billing** (pay-per-hour)
   - We recommend **HuggingFace billing** — only ~$3 total for both trainings.
   - Add a card at https://huggingface.co/billing

---

## Step 3 — Pick your GPU path

Pick ONE:

### Option A: HuggingFace GPU Space (RECOMMENDED — cheapest) ✅
- Cost: ~$0.80/hr for an L4 GPU, ~$3 total for both trainings
- Pros: runs right next to our env, no data transfer needed
- Steps in **Part A** below

### Option B: Colab Pro ($10/month subscription)
- Cost: $10/month — may be convenient if you already have it
- Pros: familiar Jupyter interface
- Steps in **Part B** below

### Option C: Local (your laptop)
- Only works if your GPU has **24GB VRAM or more** (most laptops don't)
- Skip unless you have a workstation-class GPU

---

## PART A — Running on HuggingFace GPU Space

### A.1 — Spin up a GPU Jupyter Space (2 min)
1. Go to https://huggingface.co/new-space
2. Name: `smb-training-notebook` (anything)
3. SDK: choose **Jupyter Notebook**
4. Hardware: choose **Nvidia L4 small** ($0.80/hr)
5. Visibility: Private
6. Click **Create Space**
7. Wait for it to boot (~60 sec)

### A.2 — Upload files (3 min)
In the Jupyter interface (top-right "Open in JupyterLab"):
1. Upload `training/sft_warm_start.py`
2. Upload `training/sft_data.jsonl`
3. Upload `training/train_grpo.py`
4. Upload `training/requirements-training.txt`

### A.3 — Install dependencies (5 min)
Open a terminal in the Jupyter Space (File → New → Terminal), run:
```bash
pip install -r requirements-training.txt
pip install jupytext
jupytext --to notebook sft_warm_start.py
jupytext --to notebook train_grpo.py
```
This creates `sft_warm_start.ipynb` and `train_grpo.ipynb` — open these.

### A.4 — Log into W&B
```bash
wandb login
```
Paste your W&B API key when asked.

### A.5 — Run SFT warm-start (30–45 min)
1. Open `sft_warm_start.ipynb`
2. Run all cells top to bottom
3. Final cell should show a test generation (the trained model's first response)
4. The training saves automatically to `./checkpoints/sft_adapter/`

### A.6 — Run GRPO training (45 min – 2 hours)
1. Open `train_grpo.ipynb`
2. In the first "Config" cell, confirm `SPACE_URL = "https://falgunisharma-smb-ad-manager.hf.space"`
3. Run all cells top to bottom
4. **Watch W&B** (it'll open in your browser) — you'll see the reward climbing up

### A.7 — You're done! Go to Step 4.

---

## PART B — Running on Colab Pro

### B.1 — Upload files to Google Drive (5 min)
1. Open https://drive.google.com → create a folder `smb-training`
2. Upload: `sft_warm_start.py`, `sft_data.jsonl`, `train_grpo.py`, `requirements-training.txt`

### B.2 — Open Colab with L4 GPU (1 min)
1. Go to https://colab.research.google.com → New Notebook
2. Runtime → Change runtime type → **L4 GPU** → Save

### B.3 — Install + convert (5 min)
In the Colab notebook, run these cells:

```python
# Cell 1 — mount drive
from google.colab import drive
drive.mount('/content/drive')
```

```python
# Cell 2 — copy files + install deps
!cp /content/drive/MyDrive/smb-training/*.py .
!cp /content/drive/MyDrive/smb-training/sft_data.jsonl .
!cp /content/drive/MyDrive/smb-training/requirements-training.txt .
!pip install -q -r requirements-training.txt
!pip install -q jupytext
```

### B.4 — Log into W&B
```python
# Cell 3
import wandb
wandb.login()     # paste your W&B key when prompted
```

### B.5 — Run SFT warm-start (30–45 min)
```python
# Cell 4
!jupytext --to notebook sft_warm_start.py
```
Now in the Colab file browser (left sidebar), open `sft_warm_start.ipynb` → Run all cells.

### B.6 — Run GRPO training (45 min – 4 hrs on L4)
```python
# Cell N
!jupytext --to notebook train_grpo.py
```
Open `train_grpo.ipynb` → Run all cells.

---

## Step 4 — Save + share results (15 min) — DO NOT SKIP

### 4.1 — Screenshot the W&B reward curves
1. Go to https://wandb.ai/your-username/smb-ad-manager
2. Open the GRPO run (latest one)
3. Take screenshots of these charts:
   - **Total reward over steps** (main reward curve)
   - **r1_roas_improvement over steps**
   - **r2 / r3 / r4 / r5 columns**
4. Save these to a folder named `reward_curves/`

### 4.2 — Push trained model to HF Hub (optional but recommended)
In the notebook's last cell (already there, just uncomment):
```python
model.push_to_hub("Falgunisharma/smb-ad-manager-grpo", private=True)
tokenizer.push_to_hub("Falgunisharma/smb-ad-manager-grpo", private=True)
```

### 4.3 — Download the final checkpoint (backup)
From Jupyter/Colab file browser:
1. Right-click `./checkpoints/grpo_final/`
2. Download as .zip
3. Save it somewhere safe

### 4.4 — Tell Falguni you're done
Send her:
- ✅ **The 4 screenshots** (reward curves)
- ✅ **Link to the W&B run** (URL)
- ✅ **HF Hub model name** (if you pushed it)
- ✅ **Any issues you hit**

---

## Step 5 — If something breaks

### "Out of memory" error during training
- You're on too small a GPU. Switch to L4 24GB or A10G.
- Or reduce `PER_DEVICE_BATCH_SIZE = 1` in the config cell.

### "Connection error" calling SPACE_URL
- The HF Space may have cold-started. Open https://falgunisharma-smb-ad-manager.hf.space in a browser, wait 30 sec for it to wake up, then retry.

### "Model not found" for Llama
- You need to accept Meta's Llama license: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- Then `huggingface-cli login` with your HF token.

### Training is super slow (>6 hours on L4)
- Reduce `TOTAL_STEPS = 100` (down from 200) in `train_grpo.py` config cell. Still shows improvement, just smaller curve.

### Reward curve is flat (not improving)
- Tell Falguni immediately — don't waste compute.

---

## TL;DR

```
1. Git clone the repo
2. Make HF + W&B accounts
3. Use HuggingFace GPU Space (L4, $0.80/hr)
4. Upload 4 files, install deps, wandb login
5. Run sft_warm_start.ipynb (30-45 min)
6. Run train_grpo.ipynb (45 min - 2 hrs)
7. Screenshot W&B reward curves
8. Push model to Hub + tell Falguni
```

Total cost: ~$3
Total time (yours): ~30 min active + waiting

---

**Questions?** Message Falguni on WhatsApp. Good luck! 🚀
