import torch
import os

checkpoint_path = 'sewer_checkpoint.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Batch Index: {checkpoint.get('batch_idx', 'N/A')}")
    print(f"Best F1: {checkpoint.get('best_f1', 'N/A')}")
else:
    print("Checkpoint not found.")
