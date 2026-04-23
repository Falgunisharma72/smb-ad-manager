# Training Runbook — SMB Ad Manager

**Scaler OpenEnv Hackathon · April 2026**

This document tells you everything you need to train the model end-to-end. Follow every step in order. Nobody else needs to help.

- **Total active time:** ~45 min of hands-on work
- **Total wait time:** ~2–3 hours of training
- **Total cost:** ~$3–5 on HuggingFace GPU
- **Final output:** a trained model on HuggingFace Hub + reward-curve screenshots committed back to the project repo

---

## Part 1 — Create all accounts (20 min, one-time)

Do these 6 steps in order. Each one is a prerequisite for the next.

### Step 1-1. Create a HuggingFace account
1. Open https://huggingface.co/join
2. Sign up with email
3. Verify your email via the link HF sends you
4. **Expected outcome:** You can log in and see your profile at `huggingface.co/YOUR-USERNAME`

### Step 1-2. Create a HuggingFace access token (Write scope)
1. Go to https://huggingface.co/settings/tokens
2. Click **"Create new token"**
3. Name: `smb-training`
4. Type: **Write** (NOT fine-grained — Write is simpler)
5. Click **Create**
6. **Copy the token now** (starts with `hf_...`) and save it somewhere safe — you can't see it again after you close the dialog
7. **Expected outcome:** A token string stored in your notes, visible in the tokens page

### Step 1-3. Accept the Llama 3.2 license
1. Open https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
2. You'll see a gated-model banner with a form
3. Fill it (takes 1 minute) — select a reason like "research"
4. Submit
5. **Expected outcome:** Approval usually comes in under 5 minutes. You'll get an email, and the page will say "You have been granted access to this model"
6. **Do not skip this step** — training will fail to download the base model without it

### Step 1-4. Create a Weights & Biases account (free)
1. Open https://wandb.ai/signup
2. Sign up with email or GitHub
3. **Expected outcome:** You can log in and land on your W&B home page

### Step 1-5. Get the W&B API key
1. Open https://wandb.ai/authorize
2. Copy the long key shown
3. Save it somewhere safe
4. **Expected outcome:** A key string (no prefix, ~40 characters) saved in your notes

### Step 1-6. Add a payment method on HuggingFace (pay-per-hour GPU)
1. Open https://huggingface.co/billing
2. Click **"Add payment method"**
3. Add a card (you will only be charged for what you use — ~$3)
4. **Expected outcome:** A card shown as default payment. Your account can now use paid GPU Spaces

---

## Part 2 — Clone the repo (5 min)

You have two choices for where to run training. Pick one:

- **Option A (recommended): Everything inside a HuggingFace Jupyter Space** — don't need your local machine at all
- **Option B: VS Code locally with remote kernel** — if you prefer VS Code UX

### Option A — HF Jupyter Space (easier, no local setup)

#### Step 2A-1. Create a GPU Jupyter Space
1. Open https://huggingface.co/new-space
2. Owner: your HF username
3. Space name: `smb-training-notebook`
4. License: MIT (any is fine)
5. SDK: **Jupyter Notebook**
6. Hardware: **Nvidia L4 small** ($0.80/hr)
7. Visibility: **Private**
8. Click **Create Space**
9. Wait ~90 seconds for it to boot. Status will flip from "Building" to "Running"
10. **Expected outcome:** A page showing your Space with a "Open in JupyterLab" button on the top right

#### Step 2A-2. Open a terminal inside the Space
1. Click **"Open in JupyterLab"**
2. Once JupyterLab loads, click **File → New → Terminal**
3. **Expected outcome:** A shell prompt inside the running Space

#### Step 2A-3. Clone the project repo
In that terminal, run:
```bash
cd ~
git clone https://github.com/Falgunisharma72/smb-ad-manager.git
cd smb-ad-manager
ls training/
```
**Expected outcome:** Should show:
```
README.md   generate_sft_data.py  requirements-training.txt  sft_data.jsonl
sft_warm_start.py  train_grpo.py
```

### Option B — VS Code locally with remote Jupyter

#### Step 2B-1. Clone locally
```bash
git clone https://github.com/Falgunisharma72/smb-ad-manager.git
cd smb-ad-manager
```

#### Step 2B-2. Create the HF Jupyter Space (same as Option A Step 2A-1)
You still need a GPU box somewhere. Create the Space as in Step 2A-1.

#### Step 2B-3. Connect VS Code to the Space's Jupyter server
1. Install the Jupyter extension in VS Code
2. In the HF Space, get the Jupyter URL with token from the Space's settings page
3. In VS Code: `Cmd/Ctrl+Shift+P` → **"Jupyter: Specify local or remote Jupyter server"**
4. Paste the URL
5. **Expected outcome:** VS Code shows notebooks can run on the remote kernel

> **Most people: use Option A.** It's less moving parts.

---

## Part 3 — Install dependencies (5 min)

From this point forward I'll assume you're in the HF Space's JupyterLab terminal (Option A). Same commands work for Option B.

### Step 3-1. Install Python packages
In the terminal:
```bash
cd ~/smb-ad-manager
pip install -r training/requirements-training.txt
pip install jupytext
```
**Expected outcome:** Lots of scrolling installs. Takes ~3 minutes. Final line should say `Successfully installed ...`. No red errors.

### Step 3-2. Log into HuggingFace CLI
```bash
huggingface-cli login
```
1. Paste your HF token (from Step 1-2) when prompted
2. When asked "Add token as git credential?" → **y**
3. **Expected outcome:** Message `Login successful`

### Step 3-3. Log into Weights & Biases
```bash
wandb login
```
1. Paste your W&B API key (from Step 1-5)
2. **Expected outcome:** Message `Appending key for api.wandb.ai to your netrc file`

### Step 3-4. Convert the training scripts to notebooks
```bash
jupytext --to notebook training/sft_warm_start.py
jupytext --to notebook training/train_grpo.py
```
**Expected outcome:** Two new files created: `training/sft_warm_start.ipynb` and `training/train_grpo.ipynb`. Check with `ls training/`.

---

## Part 4 — Run SFT warm-start (30–45 min)

This teaches the small Llama model the right JSON format before we do RL.

### Step 4-1. Open the notebook
1. In JupyterLab's left file browser, navigate to `smb-ad-manager/training/`
2. Double-click `sft_warm_start.ipynb`
3. **Expected outcome:** The notebook opens with ~8 cells visible

### Step 4-2. Run all cells
1. Click the **menu: Run → Run All Cells**
2. Watch the cells execute top to bottom

### Step 4-3. What should happen as each cell runs

| Cell | Expected output |
|---|---|
| 1 — Install (commented) | Skipped (we already installed) |
| 2 — Config | Prints `Config: model=... epochs=3 batch=8` |
| 3 — Load model | Downloads Llama 3.2 1B (~2 GB). First time takes 2–3 min. Shows Unsloth banner |
| 4 — Load data | `Loaded 100 examples` |
| 5 — Format | Shows a sample prompt |
| 6 — Train | W&B dashboard link printed. Progress bar shows loss decreasing over 3 epochs (~20–30 min) |
| 7 — Sanity check | Prints a JSON sample — should look like `{"tool": "update_budget", "args": {...}, "reasoning": "..."}` |

**If Cell 7 output is gibberish (not valid JSON):** SFT didn't converge. Add `EPOCHS = 5` in Cell 2 and re-run from Cell 6 only.

### Step 4-4. Verify the adapter was saved
In a terminal, check:
```bash
ls ~/smb-ad-manager/training/checkpoints/sft_adapter/
```
**Expected outcome:** Files `adapter_config.json`, `adapter_model.safetensors`, `tokenizer.json`, etc. (~10–20 MB total)

---

## Part 5 — Run GRPO training (45 min–2 hours)

This is the reinforcement-learning stage where the model learns by interacting with the live HF Space environment.

### Step 5-1. Warm up the environment Space
Before starting training, open this in a browser tab:
```
https://falgunisharma-smb-ad-manager.hf.space
```
Wait for it to load (cold starts take ~60 sec). **Leave the tab open** — this keeps the Space warm during training.

### Step 5-2. Open and start the GRPO notebook
1. In JupyterLab, open `training/train_grpo.ipynb`
2. Run the Config cell (Cell 2) — confirm these values:
   - `SFT_CHECKPOINT = "./checkpoints/sft_adapter"`
   - `SPACE_URL` = `https://falgunisharma-smb-ad-manager.hf.space`
   - `TOTAL_STEPS = 200`
   - `NUM_GENERATIONS = 2`
3. Run all remaining cells: **Run → Run All Cells**

### Step 5-3. What to watch — the reward curve

Once Cell 6 (GRPO training) starts, a **W&B dashboard link** is printed in the output. Click it and open the charts tab.

**Healthy reward curve shape:**
- Step 0–20: reward ~0.3–0.5 (baseline)
- Step 20–80: reward climbs steadily to ~0.6–0.8
- Step 80–200: reward plateaus around 0.7–0.9 with small wiggles

**Check the curve every 20–25 steps.** Don't babysit every step — step checkpoints print ~every 5 steps.

### Step 5-4. IF the reward curve goes flat (no improvement for 30+ steps)

This is the most important thing to watch. Here's what to do **while training is still running**, in priority order:

#### Fix A — Verify the environment is responding (takes 30 sec)
In a new terminal cell:
```python
import requests
r = requests.post("https://falgunisharma-smb-ad-manager.hf.space/reset",
                  json={"task_id": "easy"}, timeout=30)
print(r.status_code, r.json().get("step"))
```
**Expected:** `200 0`. If 404 / 500 / timeout → the Space is dead. Wait 60 sec, retry. If still dead, the problem is the backend; log what you see and move to Fix E.

#### Fix B — Verify reward function is producing varied scores (takes 1 min)
```python
# Quick reward-variance check
import requests, json
scores = []
for seed in range(10):
    r = requests.post("https://falgunisharma-smb-ad-manager.hf.space/reset",
                      json={"task_id": "easy", "seed": seed})
    requests.post("https://falgunisharma-smb-ad-manager.hf.space/step", json={
        "action": {"tool": "noop", "args": {}, "reasoning": "test"}
    })
    s = requests.post("https://falgunisharma-smb-ad-manager.hf.space/step", json={
        "action": {"tool": "get_metrics", "args": {"campaign_id": "c001"}, "reasoning": "test"}
    }).json()
    scores.append(s["reward"]["score"])
print("Scores:", scores, "Variance:", max(scores) - min(scores))
```
**Expected:** scores like `[0.5, 0.75, 0.9, 0.6, ...]`, variance > 0.3. If all identical, the reward isn't discriminating — jump to Fix D.

#### Fix C — Stop training, lower the learning rate, restart
1. In the GRPO notebook: **Kernel → Interrupt Kernel** (stops training)
2. Edit Cell 2 of `train_grpo.ipynb`: change `LEARNING_RATE = 1e-5` to `LEARNING_RATE = 5e-6`
3. Re-run from Cell 6 (just the training cell)
4. **Expected outcome:** Reward starts moving again within 20–30 steps

#### Fix D — Increase temperature for more exploration
1. Kernel → Interrupt
2. In Cell 6 (GRPOTrainer config), change `temperature=0.7` to `temperature=1.0`
3. Re-run from Cell 6
4. **Expected outcome:** More variance in outputs → reward function has more to work with

#### Fix E — If all else fails: switch to DPO fallback
GRPO is finicky on small models. DPO is sturdier. At the top of `train_grpo.ipynb` Cell 6, replace the GRPOTrainer setup with:
```python
# DPO FALLBACK — if GRPO keeps being flat
from trl import DPOConfig, DPOTrainer

# You'll need a preference dataset; use the SFT data as "chosen" responses
# and randomly perturbed versions as "rejected". Rough fallback:
# Skip DPO complexity for now — re-run SFT with EPOCHS=10 and submit the SFT-only model.
```
If you reach Fix E, just re-run `sft_warm_start.ipynb` with `EPOCHS = 10` and use that model as the final. It's not as strong but it's a working submission.

### Step 5-5. Final expected state
When training finishes (at `TOTAL_STEPS = 200`):
- W&B dashboard shows final reward curves for all 5 components
- Folder `./checkpoints/grpo_final/` is created locally with adapter weights
- Console prints `✓ GRPO training complete`

---

## Part 6 — Save results + push everything back to the repo (20 min)

### Step 6-1. Screenshot 4 graphs from W&B
In your W&B dashboard:
1. **Chart 1:** "rewards/reward" (the total reward curve)
2. **Chart 2:** "rewards/r1_roas_improvement"
3. **Chart 3:** "rewards/r3_format_compliance"
4. **Chart 4:** "rewards/r5_no_cheating"

For each: right-click on the chart → **"Save image"** as PNG. Name them:
- `reward_curve_total.png`
- `reward_curve_r1.png`
- `reward_curve_r3.png`
- `reward_curve_r5.png`

### Step 6-2. Upload screenshots into the project
In JupyterLab terminal:
```bash
cd ~/smb-ad-manager
mkdir -p results
```

Upload the 4 PNGs via drag-and-drop into the `results/` folder in JupyterLab's file browser.

### Step 6-3. Create a training summary file
Create a new text file `results/training_summary.md` with this content (replace values):
```markdown
# Training Run Summary

**Run date:** 2026-04-XX
**Run by:** <your name>
**GPU used:** HuggingFace L4
**Total cost:** $X.XX

## SFT warm-start
- Model: Llama 3.2 1B Instruct
- Examples: 100
- Epochs: 3 (or 5 if you bumped)
- Final loss: <check W&B>

## GRPO training
- Algorithm: 2-GRPO (group size 2)
- Steps: 200
- Starting reward (step 0): 0.XX
- Final reward (step 200): 0.XX
- Best reward: 0.XX

## W&B run URL
https://wandb.ai/<your-username>/smb-ad-manager/runs/<run-id>

## Trained model on HF Hub
https://huggingface.co/<your-username>/smb-ad-manager-grpo
```

### Step 6-4. Push the trained model to HuggingFace Hub
In a notebook cell (at the bottom of `train_grpo.ipynb`), uncomment and run:
```python
model.push_to_hub("YOUR-HF-USERNAME/smb-ad-manager-grpo", private=True)
tokenizer.push_to_hub("YOUR-HF-USERNAME/smb-ad-manager-grpo", private=True)
```
Replace `YOUR-HF-USERNAME` with your actual HF username (e.g., `Sarthak123` — whatever yours is).

**Expected outcome:** Progress bar shows ~20 MB upload. Ends with a URL like `https://huggingface.co/YOUR-HF-USERNAME/smb-ad-manager-grpo`.

### Step 6-5. Commit + push results to the GitHub repo
Back in the terminal:
```bash
cd ~/smb-ad-manager
git config user.name "YOUR NAME"
git config user.email "your.email@example.com"

git add results/
git commit -m "results: training run complete — reward curves + summary"

# Push to GitHub:
git push origin main
```
**Expected outcome:** The 4 PNGs and `training_summary.md` appear on GitHub at `github.com/Falgunisharma72/smb-ad-manager/tree/main/results`.

### Step 6-6. Also push to the HuggingFace Space mirror
The project has a second git remote for the HF Space:
```bash
git push hf main
```
If that remote doesn't exist, add it:
```bash
git remote add hf https://YOUR-HF-USERNAME:YOUR-HF-TOKEN@huggingface.co/spaces/Falgunisharma/smb-ad-manager
git push hf main
```
**Expected outcome:** HF Space rebuilds (takes ~60s). Doesn't matter if it changes behavior — we just want the results/ folder mirrored there.

### Step 6-7. Verify everything is where it should be
- GitHub repo → `results/` folder has 5 files (4 PNGs + summary) ✓
- HF Hub → `YOUR-USERNAME/smb-ad-manager-grpo` exists and is browsable ✓
- W&B run is visible in your W&B dashboard and not deleted ✓

---

## Part 7 — Shut down the GPU (SAVE MONEY)

**This is important — the GPU bills per hour even when idle.**

1. Go to `huggingface.co/spaces/YOUR-USERNAME/smb-training-notebook/settings`
2. Scroll to **"Sleep time"** or **"Pause"**
3. Either:
   - Click **"Pause space"** (stops billing immediately, can restart later)
   - OR change hardware to **"CPU basic"** (free tier; keeps files)
   - OR **"Delete this Space"** if you're 100% done and have pushed everything

**Expected outcome:** GPU billing stops. Confirmed by billing page showing "Not running."

---

## Part 8 — Final checklist

Before walking away, confirm ALL of these:

- [ ] SFT adapter is at `training/checkpoints/sft_adapter/` in the repo (committed)
- [ ] GRPO adapter is pushed to HuggingFace Hub at `YOUR-HF-USERNAME/smb-ad-manager-grpo`
- [ ] 4 reward curve PNGs are in `results/` folder on GitHub
- [ ] `results/training_summary.md` has real numbers in it
- [ ] W&B run URL is public (or at least sharable within your team)
- [ ] GPU Space is paused or deleted
- [ ] You have a backup download of the GRPO checkpoint on your laptop (zip the folder and download it as insurance)

---

## Part 9 — Full troubleshooting reference

### Training-time issues

| Symptom | Cause | Fix |
|---|---|---|
| Reward curve flat for 30+ steps | GRPO not exploring enough OR env broken | Run the 30-sec and 1-min checks in Step 5-4 Fix A and B. Then try Fix C or D. |
| Reward curve going DOWN | Learning rate too high | Interrupt, set `LEARNING_RATE = 5e-6`, restart from Cell 6 |
| OOM (out of memory) | Model too big for GPU | Change `PER_DEVICE_BATCH_SIZE = 1` in Cell 2 |
| `403 Forbidden` on Llama download | License not accepted | Do Step 1-3 again, wait for approval email |
| `Connection refused` to SPACE_URL | HF Space cold-started or restarted | Open the Space URL in a browser, wait 60s, retry |
| Cell 7 sanity check prints non-JSON | SFT didn't converge | Set `EPOCHS = 5` in Cell 2, re-run Cell 6 onward |
| Training runs 5+ hours with no end | Too many steps for this GPU | Interrupt, change `TOTAL_STEPS = 100`, use what you have |

### Git push issues

| Symptom | Fix |
|---|---|
| `Rejected — fetch first` | Run `git pull origin main --rebase`, then push again |
| `Permission denied` pushing to HF | Get a new Write-scope HF token, `huggingface-cli login` again |
| Adapter files not uploading to Hub | Check that the folder contains `adapter_model.safetensors`. If missing, re-run the `push_to_hub` cell. |

---

## Quick reference — key URLs to keep open

- GitHub repo: https://github.com/Falgunisharma72/smb-ad-manager
- Live env (warm before training): https://falgunisharma-smb-ad-manager.hf.space
- Your training Space: https://huggingface.co/spaces/YOUR-USERNAME/smb-training-notebook
- Your W&B project: https://wandb.ai/YOUR-USERNAME/smb-ad-manager
- Your trained model (after push): https://huggingface.co/YOUR-USERNAME/smb-ad-manager-grpo

---

## TL;DR (if you're in a hurry)

1. Make accounts: HF + W&B + HF billing + Llama license (Part 1)
2. Create HF Jupyter Space with L4 GPU (Part 2)
3. Clone repo, install deps, log into HF + W&B (Part 3)
4. Run `sft_warm_start.ipynb` top to bottom (Part 4)
5. Warm up the env Space in a browser tab, then run `train_grpo.ipynb` (Part 5)
6. Watch W&B charts; apply Fix A/B/C/D if reward goes flat (Part 5-4)
7. Screenshot 4 charts, write `results/training_summary.md`, push checkpoint to HF Hub (Part 6)
8. Commit everything to the GitHub repo — `git add results/ && git commit && git push origin main && git push hf main`
9. **Pause or delete the GPU Space** (Part 7)
10. Confirm the final checklist (Part 8)

Total active time: ~45 min. Total wait: ~2.5 hrs. Total cost: ~$3.
