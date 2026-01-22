import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models
from dataset import SewerDataset, get_transforms
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
import os
import json

def optimize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    csv_path = 'train.csv'
    img_dir = 'train_images'
    
    # We use augment=False for validation/optimization
    transform = get_transforms(augment=False)
    full_dataset = SewerDataset(csv_path, img_dir, transform=transform)
    
    # Recreate the split exactly as in training
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Validation set size: {len(val_dataset)}")
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=4,
        pin_memory=(device.type == 'cuda')
    )

    # Load Model
    model_path = "sewer_model_fixed.pth"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    print("Loading model...")
    model = models.efficientnet_b0(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    # 19 classes
    model.classifier[1] = nn.Linear(num_ftrs, 19)
    
    print(f"Loading state dict from {model_path}...")
    try:
        state_dict = torch.load(model_path, map_location=device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            
        model.load_state_dict(state_dict, strict=True)
        print("Model loaded successfully (strict=True).")
    except Exception as e:
        print(f"Strict load failed: {e}")
        print("Attempting strict=False load...")
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"Missing keys: {len(missing)}")
            print(f"Unexpected keys: {len(unexpected)}")
            # print(f"First 5 missing: {missing[:5]}")
            # print(f"First 5 unexpected: {unexpected[:5]}")
        except Exception as e2:
             print(f"Fatal error loading model: {e2}")
             return
    model = model.to(device)
    model.eval()

    # Run Inference on Validation Set
    all_labels = []
    all_preds = []
    
    print("Running inference on validation set...")
    max_batches = 50
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(val_loader, total=max_batches)):
            if i >= max_batches:
                break
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds_probs = np.array(all_preds)
    all_labels = np.array(all_labels)

    # optimize thresholds
    print("Optimizing thresholds...")
    best_thresholds = []
    labels_list = ['VA', 'RB', 'OB', 'PF', 'DE', 'FS', 'IS', 'RO', 'IN', 'AF', 'BE', 'FO', 'GR', 'PH', 'PB', 'OS', 'OP', 'OK', 'ND']
    
    for i in range(19): 
        class_preds = all_preds_probs[:, i] 
        class_labels = all_labels[:, i]
        best_t = 0.5
        best_class_f1 = 0.0
        
        # Search range
        for t in np.arange(0.05, 1.0, 0.05):
            t_preds = (class_preds > t).astype(int)
            t_f1 = f1_score(class_labels, t_preds, zero_division=0)
            if t_f1 > best_class_f1:
                best_class_f1 = t_f1
                best_t = t
        
        best_thresholds.append(float(best_t))
        print(f"Class {labels_list[i]}: Best Threshold={best_t:.2f}, F1={best_class_f1:.4f}")

    # Calculate final Macro F1 with optimized thresholds
    final_preds = np.zeros_like(all_preds_probs)
    for i in range(19):
        final_preds[:, i] = (all_preds_probs[:, i] > best_thresholds[i]).astype(int)
        
    macro_f1 = f1_score(all_labels, final_preds, average='macro')
    print(f"Final Validation Macro F1: {macro_f1:.4f}")

    # Save to JSON
    with open('thresholds.json', 'w') as f:
        json.dump(best_thresholds, f)
    
    print("Saved optimized thresholds to thresholds.json")

if __name__ == "__main__":
    optimize()
