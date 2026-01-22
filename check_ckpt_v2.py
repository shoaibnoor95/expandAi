import torch
import os

def check_checkpoint(path):
    if not os.path.exists(path):
        print(f"Skipping {path}: File not found")
        return

    try:
        print(f"--- Checking {path} ---")
        # Load on CPU to avoid CUDA errors if on non-GPU environment
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        
        if isinstance(checkpoint, dict):
            if 'epoch' in checkpoint:
                print(f"Epoch: {checkpoint['epoch']}")
            if 'best_f1' in checkpoint:
                print(f"Best F1 Score: {checkpoint['best_f1']:.4f}")
            if 'batch_idx' in checkpoint:
                print(f"Batch Index: {checkpoint['batch_idx']}")
        else:
            print("Checkpoint is not a dictionary (likely just state_dict).")
            
    except Exception as e:
        print(f"Error reading {path}: {e}")

if __name__ == "__main__":
    check_checkpoint("sewer_checkpoint.pth")
    check_checkpoint("sewer_model_best.pth")
