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
python train_model.py
```

### Useful Flags
The `train_model.py` script has default arguments, but you can modify them in the `if __name__ == "__main__":` block or by modifying the parameters:

- `epochs`: Number of training epochs (default: 5)
- `batch_size`: Batch size (default: 48, you can likely increase this on H200)
- `subset_size`: Set to 0 for full dataset, or a number for testing (default: 0)

To run with modifications, you can edit the command in `train_model.py` or run a command like:
```bash
# Example of passing args if script supported them via CLI (currently need to edit file or add CLI args)
python train_model.py
```
*Currently `train_model.py` runs `train_model(epochs=5, batch_size=48, subset_size=0, resume=True)` at the bottom. You can edit this line in the file to change parameters for the H200 (e.g., `batch_size=128` or `256` to utilize the memory).*

## 6. Monitor Progress
The script logs to `training.log` (or similar logs defined in code) and uses `tqdm` for a progress bar.
- `sewer_model.pth`: Final model
- `sewer_model_best.pth`: Best model based on F1 score
- `sewer_checkpoint.pth`: Resume checkpoint

## 7. Inference
To generate the submission file:
```bash
python inference.py
```
This will produce `submission.csv`.
