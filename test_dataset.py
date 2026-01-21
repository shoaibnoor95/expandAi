from dataset import SewerDataset, get_transforms
import torch
import os

def test_dataset():
    csv_path = 'train.csv'
    img_dir = 'train_images'
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    if not os.path.exists(img_dir):
        print(f"Error: {img_dir} not found.")
        return

    transform = get_transforms()
    dataset = SewerDataset(csv_path, img_dir, transform=transform)
    
    print(f"Dataset length: {len(dataset)}")
     
    # Try loading the first 5 samples
    for i in range(5):
        img, label = dataset[i]
        print(f"Sample {i}: Image shape: {img.shape}, Labels type: {type(label)}, Labels: {label}")

if __name__ == "__main__":
    test_dataset()
