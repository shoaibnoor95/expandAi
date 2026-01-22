import torch
import torch.nn as nn
from torchvision import models
import os

def debug_load():
    device = "cpu"
    print(f"Using device: {device}")

    model_path = "sewer_model_best.pth"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    print("Building model (EfficientNet-B0, 19 classes)...")
    model = models.efficientnet_b0(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 19)
    
    print("Loading state dict...")
    try:
        state_dict = torch.load(model_path, map_location=device)
        
        # Check if wrapped in dict
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            
        dataset_res = model.load_state_dict(state_dict, strict=True)
        print("Success!")
    except RuntimeError as e:
        print(f"Caught RuntimeError: {e}")

if __name__ == "__main__":
    debug_load()
