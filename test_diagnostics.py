
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import os
from dataset import get_transforms

def test_single():
    torch.backends.mkldnn.enabled = False
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    labels = ['VA', 'RB', 'OB', 'PF', 'DE', 'FS', 'IS', 'RO', 'IN', 'AF', 'BE', 'FO', 'GR', 'PH', 'PB', 'OS', 'OP', 'OK', 'ND']
    model = models.efficientnet_b0(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(labels))
    
    model_path = "sewer_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    
    model.eval()
    
    # Create dummy input
    transform = get_transforms()
    dummy_input = torch.randn(1, 3, 224, 224)
    
    print("Running inference on dummy input...")
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print("Success!")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_single()
