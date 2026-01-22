import torch
import os

model_path = "sewer_model_best.pth"
if os.path.exists(model_path):
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        # Check if it's a full checkpoint or just state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("Loaded from checkpoint['model_state_dict']")
        else:
            state_dict = checkpoint
            print("Loaded from raw state_dict")
            
        for key in state_dict:
            if 'classifier' in key:
                print(f"{key}: {state_dict[key].shape}")
    except Exception as e:
        print(f"Error loading: {e}")
else:
    print("Model file not found")
