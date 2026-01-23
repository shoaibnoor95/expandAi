import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from PIL import Image
import pandas as pd
import os
from dataset import get_transforms
from tqdm import tqdm
import numpy as np

class SewerTestDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.df.iloc[idx, 0])
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_name}: {e}")
            # Return a blank image if error
            image = Image.new('RGB', (224, 224))
        
        # Test Time Augmentation (TTA): Original, HFlip, VFlip
        images = [image]
        images.append(image.transpose(Image.FLIP_LEFT_RIGHT))
        images.append(image.transpose(Image.FLIP_TOP_BOTTOM))

        if self.transform:
            images = [self.transform(img) for img in images]
            
        # Stack into [3, C, H, W]
        return torch.stack(images), self.df.iloc[idx, 0]

def run_inference(model_path="sewer_model_best.pth", test_csv="test.csv", img_dir="test_images", output_csv="submission.csv", subset_size=0):
    torch.backends.mkldnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Labels order from train.csv
    labels = ['VA', 'RB', 'OB', 'PF', 'DE', 'FS', 'IS', 'RO', 'IN', 'AF', 'BE', 'FO', 'GR', 'PH', 'PB', 'OS', 'OP', 'OK', 'ND']
    
    # Model
    model = models.efficientnet_b4(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(labels))
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: model file {model_path} not found. Using untrained model.")
        
    model = model.to(device)
    model.eval()

    # Dataset
    transform = get_transforms()
    test_dataset = SewerTestDataset(test_csv, img_dir, transform=transform)
    
    if subset_size > 0:
        # Use only first few for demonstration
        test_dataset.df = test_dataset.df.head(subset_size)
        print(f"Running inference on subset of {subset_size} images")

    # Check for existing results to resume
    columns = ["Filename"] + labels
    processed_filenames = set()
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        try:
            # Only read Filename column to save memory
            existing_df = pd.read_csv(output_csv, usecols=["Filename"])
            processed_filenames = set(existing_df['Filename'].tolist())
            print(f"Resuming: {len(processed_filenames)} images already processed.")
        except Exception as e:
            print(f"Could not resume from {output_csv}: {e}")
            pd.DataFrame(columns=columns).to_csv(output_csv, index=False)
    else:
        pd.DataFrame(columns=columns).to_csv(output_csv, index=False)

    # Filter dataset to only unprocessed
    if len(processed_filenames) > 0:
        test_dataset.df = test_dataset.df[~test_dataset.df.iloc[:, 0].isin(processed_filenames)].reset_index(drop=True)
        print(f"Remaining images to process: {len(test_dataset.df)}")

    if len(test_dataset.df) == 0:
        print("All images already processed.")
        return

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    import time
    buffer = []
    buffer_size = 50 # Write every 50 batches (800 images) to reduce lock frequency
    
    with torch.no_grad():
        for i, (inputs, filenames) in enumerate(tqdm(test_loader, desc="Inference (TTA)")):
            # Load thresholds if available
            thresholds = 0.5
            if os.path.exists('thresholds.json'):
                import json
                with open('thresholds.json', 'r') as f:
                    thresholds_list = json.load(f)
                    thresholds = np.array(thresholds_list)
                    # print("Using optimized thresholds") # Optional: uncomment to confirm
            
            # inputs shape: [Batch, 3, C, H, W]
            bs, n_crops, c, h, w = inputs.size()
            
            # Fuse batch and crops
            inputs = inputs.view(-1, c, h, w).to(device) # [Batch*3, C, H, W]
            
            outputs = model(inputs) # [Batch*3, 19]
            
            # Reshape back to separate crops and average
            outputs = outputs.view(bs, n_crops, -1) # [Batch, 3, 19]
            probs = torch.sigmoid(outputs).mean(dim=1) # Average probabilities [Batch, 19]
            
            preds = probs.cpu().numpy()
            
            # Apply thresholds
            if isinstance(thresholds, np.ndarray):
                binary_preds = (preds > thresholds).astype(int)
            else:
                binary_preds = (preds > thresholds).astype(int)
            
            for j in range(len(filenames)):
                res = [filenames[j]] + list(binary_preds[j])
                buffer.append(res)
            
            if (i + 1) % buffer_size == 0 or (i + 1) == len(test_loader):
                # Retry loop for PermissionError
                max_retries = 5
                for retry in range(max_retries):
                    try:
                        batch_df = pd.DataFrame(buffer, columns=columns)
                        batch_df.to_csv(output_csv, mode='a', header=False, index=False)
                        buffer = [] # Clear buffer on success
                        break
                    except PermissionError:
                        if retry < max_retries - 1:
                            print(f"\nPermission denied on {output_csv}, retrying in 2s... (Attempt {retry+1}/{max_retries})")
                            time.sleep(2)
                        else:
                            print(f"\nCritical: Could not write to {output_csv} after {max_retries} attempts.")
                            # Still keep results in buffer for next successful write if possible
                            # or just continue and hope next one works

    print(f"Submission saved to {output_csv}")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Inference on Sewer-ML")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output CSV file")
    parser.add_argument("--subset-size", type=int, default=0, help="Number of images to process (0 for all)")
    parser.add_argument("--model", type=str, default="sewer_model_best.pth", help="Path to model file")
    
    args = parser.parse_args()
    
    run_inference(
        model_path=args.model,
        output_csv=args.output,
        subset_size=args.subset_size
    )

