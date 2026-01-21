import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class SewerDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.transform = transform
        self.refresh_file_list()
        
    def refresh_file_list(self):
        """Rescans the image directory and updates the dataframe to include new files."""
        existing_files = set(os.listdir(self.img_dir))
        original_df = pd.read_csv(self.csv_file)
        self.df = original_df[original_df.iloc[:, 0].isin(existing_files)].reset_index(drop=True)
        print(f"Dataset updated: {len(self.df)} images now available.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.df.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        
        labels = self.df.iloc[idx, 1:].values.astype('float32')
        labels = torch.from_numpy(labels)

        if self.transform:
            image = self.transform(image)

        return image, labels

def get_transforms(img_size=380, augment=False):
    if augment:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(), # Added vertical flip
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Added shift
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
