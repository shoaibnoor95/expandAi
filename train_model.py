import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torchvision import models
from dataset import SewerDataset, get_transforms
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
import os

def train_model(epochs=20, batch_size=256, lr=0.001, subset_size=0, resume=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Optimize for CPU training if no GPU
    cpu_cores = os.cpu_count() or 1
    if device.type == 'cpu':
        torch.set_num_threads(cpu_cores)
        print(f"Set torch num_threads to {cpu_cores}")
    
    # Increase workers for Linux/High-end CPU
    num_workers = min(16, cpu_cores)

    # Load dataset
    csv_path = 'train.csv'
    img_dir = 'train_images'
    train_transform = get_transforms(augment=True)
    val_transform = get_transforms(augment=False)
    
    full_dataset = SewerDataset(csv_path, img_dir, transform=train_transform)
    
    # Model
    print("Initializing EfficientNet-B4...")
    model = models.efficientnet_b4(pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 19)
    model = model.to(device)

    # Calculate class weights
    print("Calculating class weights...")
    df = full_dataset.df
    labels_list = ['VA', 'RB', 'OB', 'PF', 'DE', 'FS', 'IS', 'RO', 'IN', 'AF', 'BE', 'FO', 'GR', 'PH', 'PB', 'OS', 'OP', 'OK', 'ND']
    counts = df[labels_list].sum().values
    weights = 1.0 / (counts + 1)
    weights = weights / weights.sum() * len(labels_list)
    class_weights = torch.FloatTensor(weights).to(device)
    print(f"Class weights calculated.")

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    start_epoch = 0
    start_batch = 0
    best_f1 = 0.0
    checkpoint_path = "sewer_checkpoint.pth"
    saved_batch_size = 32 # Value used in previous runs

    if resume and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        
        # Adjust start_batch if batch_size changed
        old_batch_idx = checkpoint.get('batch_idx', -1)
        saved_batch_size = checkpoint.get('batch_size', 32) # Load saved batch size or default to 32
        if old_batch_idx != -1:
            samples_processed = (old_batch_idx + 1) * saved_batch_size
            start_batch = samples_processed // batch_size
        else:
            start_batch = 0
            
        best_f1 = checkpoint.get('best_f1', 0.0)
        print(f"Resumed from epoch {start_epoch+1}, batch {start_batch} (approx). Best F1: {best_f1:.4f}")

    # Training loop
    for epoch in range(start_epoch, epochs):
        full_dataset.refresh_file_list()
        
        if subset_size > 0:
            indices = np.random.choice(len(full_dataset), min(subset_size, len(full_dataset)), replace=False)
            train_indices = indices[:int(0.8 * len(indices))]
            val_indices = indices[int(0.8 * len(indices)):]
            train_dataset = Subset(full_dataset, train_indices)
            val_dataset = Subset(full_dataset, val_indices)
        else:
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size], 
                generator=torch.Generator().manual_seed(42)
            )
        
        # Efficiently skip processed samples if resuming mid-epoch
        if epoch == start_epoch and start_batch > 0:
            samples_to_skip = start_batch * batch_size
            if samples_to_skip < len(train_dataset):
                print(f"Efficiently skipping {samples_to_skip} samples for resume...")
                # We create a new subset starting from the resume point
                # This works if the dataset order is deterministic (our random_split has a fixed seed)
                actual_train_indices = range(samples_to_skip, len(train_dataset))
                current_train_dataset = Subset(train_dataset, actual_train_indices)
            else:
                current_train_dataset = train_dataset # Should not happen if start_batch is valid
        else:
            current_train_dataset = train_dataset

        train_loader = DataLoader(
            current_train_dataset, 
            batch_size=batch_size, 
            shuffle=(epoch != start_epoch or start_batch == 0), # Shuffle only if not resuming mid-epoch (to keep indexing simple)
            num_workers=num_workers, 
            pin_memory=(device.type == 'cuda')
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0, # Use 0 workers for validation stability on Windows
            pin_memory=(device.type == 'cuda')
        )
        
        print(f"Train samples: {len(current_train_dataset)}, Val samples: {len(val_dataset)}")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        print(f"Using {num_workers} workers for training, 0 for validation.")

        model.train()
        running_loss = 0.0
        
        # tqdm starts from 0 because current_train_dataset is already sliced
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
            if (i + 1) % 10 == 0:
                pbar.set_postfix({'loss': loss.item()})
            
            # Mid-epoch checkpoint (adjust batch_idx to be absolute for resume logic)
            if (i + 1) % 500 == 0:
                absolute_batch_idx = i + (start_batch if epoch == start_epoch else 0)
                torch.save({
                    'epoch': epoch,
                    'batch_idx': absolute_batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_f1': best_f1,
                    'batch_size': batch_size
                }, checkpoint_path)

        # Correct denominator for loss
        train_samples_count = len(current_train_dataset)
        epoch_loss = running_loss / train_samples_count if train_samples_count > 0 else 0
        print(f"Epoch Loss: {epoch_loss:.4f}")

        # Validation
        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        all_preds_probs = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Per-class threshold optimization
        print("Optimizing thresholds...")
        best_thresholds = []
        for i in range(19): # 19 classes
            class_preds = all_preds_probs[:, i] 
            class_labels = all_labels[:, i]
            best_t = 0.5
            best_class_f1 = 0.0
            for t in np.arange(0.1, 1.0, 0.05):
                t_preds = (class_preds > t).astype(int)
                t_f1 = f1_score(class_labels, t_preds, zero_division=0)
                if t_f1 > best_class_f1:
                    best_class_f1 = t_f1
                    best_t = t
            best_thresholds.append(best_t)
        
        best_thresholds = np.array(best_thresholds)
        print(f"Optimal thresholds: {best_thresholds}")

        # Apply optimized thresholds
        final_preds = (all_preds_probs > best_thresholds).astype(int)
        
        f1 = f1_score(all_labels, final_preds, average='macro')
        print(f"Validation F1-score (Macro) with optimized thresholds: {f1:.4f}")
        
        scheduler.step(f1)

        torch.save({
            'epoch': epoch + 1,
            'batch_idx': -1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_f1': max(f1, best_f1),
            'batch_size': batch_size
        }, checkpoint_path)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "sewer_model_best.pth")
            print(f"Best model saved with F1: {best_f1:.4f}")

    torch.save(model.state_dict(), "sewer_model.pth")
    print("Final model saved to sewer_model.pth")

if __name__ == "__main__":
    train_model(epochs=20, batch_size=256, subset_size=0, resume=True)

