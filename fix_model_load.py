import torch
import torch.nn as nn
from torchvision import models
import os

def fix_model():
    device = "cpu"
    print(f"Using device: {device}")

    model_path = "sewer_model_best.pth"
    output_path = "sewer_model_fixed.pth"
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    print("Building local model (EfficientNet-B0, 19 classes)...")
    model = models.efficientnet_b0(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 19)
    
    local_state = model.state_dict()
    
    print(f"Loading checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        ckpt_state = checkpoint['model_state_dict']
    else:
        ckpt_state = checkpoint
        
    print(f"Checkpoint keys: {len(ckpt_state)}")
    print(f"Local model keys: {len(local_state)}")
    
    new_state_dict = {}
    mismatched_keys = []
    
    for key, val in ckpt_state.items():
        if key in local_state:
            if val.shape == local_state[key].shape:
                new_state_dict[key] = val
            else:
                mismatched_keys.append((key, val.shape, local_state[key].shape))
        else:
            # Key not in local model, ignore
            pass
            
    print(f"\nFound {len(mismatched_keys)} mismatched keys.")
    if len(mismatched_keys) > 0:
        print("Mismatched keys (Checkpoint vs Local):")
        for k, S1, S2 in mismatched_keys:
            print(f" - {k}: {S1} vs {S2}")
    
    print(f"\nCompatible keys to load: {len(new_state_dict)}")
    
    print("Loading compatible keys...")
    model.load_state_dict(new_state_dict, strict=False)
    
    print(f"Saving fixed model to {output_path}...")
    torch.save(model.state_dict(), output_path)
    print("Done.")

if __name__ == "__main__":
    fix_model()
