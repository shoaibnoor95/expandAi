# Server Setup and Training Guide

This guide outlines the steps to set up your environment on the new server (NVIDIA H200) and run the training code.

## 1. Clone the Repository
Once the code is pushed to your GitHub, clone it to your server:

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd <REPO_DIRECTORY>
```

## 2. Set Up Python Environment
It is recommended to use a virtual environment or Conda.

### Option A: Using venv (Standard Python)
```bash
# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate it
# On Linux/Linux-like (likely for H200 server):
source venv/bin/activate
# On Windows:
# .\venv\Scripts\Activate
```

### Option B: Using Conda
```bash
conda create -n sewer_ml python=3.10
conda activate sewer_ml
```

## 3. Install Dependencies
Install the required packages using the `requirements.txt` file included in the repo.

```bash
pip install -r requirements.txt
```

*Note: Since you are using an NVIDIA H200, ensure you have the correct CUDA version installed. PyTorch usually comes with its own CUDA runtime, but if you run into issues, check [pytorch.org](https://pytorch.org/get-started/locally/) for the specific install command for your CUDA version (e.g., CUDA 12.1 or 12.4).*

## 4. Prepare Data
As you mentioned, download the `train_images` and `test_images` directories and place them in the root of the repository folder.

Directory structure should look like this:
```
/path/to/repo/
├── train_images/      <-- Downloaded
├── test_images/       <-- Downloaded
├── train.csv         <-- Included in repo (or download if large)
├── train_model.py
├── requirements.txt
...
```
*Ensure `train.csv` is also present. If it was too large for GitHub, download it as well.*

## 5. Run Training
To start the training script:

```bash
python3 train_model.py
```

### Optimized Command for NVIDA H200 (235GB RAM)
To utilize your server's full potential, increase the batch size and workers:
```bash
python3 train_model.py --batch-size 192 --workers 32
```

### CLI Arguments
You can now control training parameters directly from the command line:
- `--batch-size`: Set batch size (default: 64). Try 192 or 256 for H200.
- `--workers`: Number of data loading workers (default: auto). Try 32 for high-core CPU.
- `--epochs`: Number of epochs (default: 20).
- `--subset-size`: Use a subset of data for testing (default: 0 for all data).

Example:
```bash
python3 train_model.py --batch-size 192 --workers 32 --epochs 30
```

## 6. Resuming Training
The script is configured to automatically resume training if it finds a checkpoint file.

### Scenario A: Resuming a Server Run
If your training on the server was interrupted, simply run the script again:
```bash
python3 train_model.py
```
It will detect `sewer_checkpoint.pth` in the directory and continue from the last saved epoch/batch.

### Scenario B: Continue Local Training on Server
If you started training locally and want to finish on the H200:
1. Upload your local `sewer_checkpoint.pth` file to the root of the repo on the server (same level as `train_model.py`).
2. Run `python3 train_model.py`.

## 7. Monitor Progress
The script logs to `training.log` (or similar logs defined in code) and uses `tqdm` for a progress bar.
- `sewer_model.pth`: Final model
- `sewer_model_best.pth`: Best model based on F1 score
- `sewer_checkpoint.pth`: Resume checkpoint

## 9. Running in Background (SSH Safe)
To keep the training running even if you disconnect from SSH, use `nohup`:

```bash
nohup python3 train_model.py --batch-size 192 --workers 32 > training.log 2>&1 &
```

### Explaining the command:
- `nohup`: "No Hang Up" - prevents process from stopping when you logout.
- `> training.log`: Redirects output to a file named `training.log`.
- `2>&1`: Redirects errors to the same file.
- `&`: Runs the process in the background immediately.

### Checking Progress
To see the "live" output (like you would see on screen):
```bash
tail -f training.log
```
*Press `Ctrl+C` to stop watching (this won't stop the training).*

### Stopping the Process
If you need to stop the background training:
1. Find the Process ID (PID):
   ```bash
   ps aux | grep train_model.py
   ```
2. Kill the process:
   ```bash
## 10. Inference
To generate the submission file:
```bash
python inference.py
```
This will produce `submission.csv`.
