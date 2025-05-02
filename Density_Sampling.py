import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_encoder
import cv2
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA

class CellSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        image = image.astype('float32') / 255.0
        mask = (mask > 0).astype('float32')
        
        image = torch.tensor(image).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)
        
        image = pad_to_multiple(image)
        mask = pad_to_multiple(mask)
        
        return image, mask, self.image_filenames[idx]

def pad_to_multiple(x, multiple=32):
    h, w = x.shape[-2], x.shape[-1]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return F.pad(x, (0, pad_w, 0, pad_h))

train_ds = CellSegmentationDataset("data/images_train", "data/masks_train")
val_ds = CellSegmentationDataset("data/images_val", "data/masks_val")
test_ds = CellSegmentationDataset("data/images_test", "data/masks_test")

def extract_features(dataset, batch_size=16, cache_path="cached_features.npy"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        return np.load(cache_path)

    encoder = get_encoder(
        "resnet34",
        in_channels=1,
        depth=5,
        weights="imagenet"
    ).to(device)
    encoder.eval()

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    features = []

    with torch.no_grad():
        for imgs, _, _ in loader:
            imgs = imgs.to(device)
            feat = encoder(imgs)[-1] 
            pooled = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1)).squeeze(-1).squeeze(-1)  # (B, C)
            features.append(pooled.cpu().numpy())

    features_array = np.concatenate(features, axis=0)
    np.save(cache_path, features_array)
    print(f"Saved features to {cache_path}")
    return features_array

def compute_local_density(features, k=10):
    #pairwise distances
    distances = pairwise_distances(features)
    #distance to its k-th nearest neighbor
    k_distances = np.sort(distances, axis=1)[:, 1:k+1]
    #density is inversely proportional to avg distance to k nearest neighbors
    density = 1.0 / (np.mean(k_distances, axis=1) + 1e-10)
    return density

def density_diversity_sampling(features, n_samples, k=10, alpha=0.7):
    density = compute_local_density(features, k)    
    selected_indices = []
    remaining_indices = list(range(len(features)))
    
    first_idx = np.argmax(density)
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)    
    all_distances = pairwise_distances(features)
    
    for _ in range(1, n_samples):
        if len(remaining_indices) == 0:
            break
        remaining_to_selected = all_distances[remaining_indices][:, selected_indices]        
        min_distances = np.min(remaining_to_selected, axis=1)
        #normalized density scores
        density_scores = density[remaining_indices] / np.max(density)
        scores = alpha * density_scores + (1 - alpha) * min_distances
        
        best_idx_local = np.argmax(scores)
        best_idx = remaining_indices[best_idx_local]
        
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
    return np.array(selected_indices)

def evaluate_model_on_subset(dataset, subset_indices, test_loader, epochs=5):
    subset = Subset(dataset, subset_indices)
    loader = DataLoader(subset, batch_size=8, shuffle=True) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights="imagenet", 
        in_channels=1, 
        classes=1, 
        activation="sigmoid"
    ).to(device)
    
    loss_fn = smp.losses.DiceLoss(mode='binary')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    best_loss = float('inf')
    patience = 2
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs, masks, _ in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        
        #early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    #evaluation on training set
    model.eval()
    train_dice_scores = []
    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            preds_bin = (preds > 0.5).float()
            intersection = (preds_bin * masks).sum()
            union = preds_bin.sum() + masks.sum()
            dice = (2 * intersection) / (union + 1e-8)
            train_dice_scores.append(dice.item())
    
    #evaluation on test set
    model.eval()
    test_dice_scores = []
    with torch.no_grad():
        for img, mask, _ in test_loader:
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            pred_bin = (pred > 0.5).float()
            inter = (pred_bin * mask).sum()
            union = pred_bin.sum() + mask.sum()
            dice = (2 * inter) / (union + 1e-8)
            test_dice_scores.append(dice.item())
    
    return np.mean(train_dice_scores), np.mean(test_dice_scores)

def main():
    print("Extracting features for density-based sampling...")
    features = extract_features(train_ds)

    initial_size = 100
    increment = 200
    max_size = int(0.5 * len(train_ds))
    n_simulations = 5  

    dataset_sizes = list(range(initial_size, max_size + 1, increment))
    train_results, test_results = {}, {}

    test_loader = DataLoader(test_ds, batch_size=4)

    for sim in range(n_simulations):
        print(f"Starting simulation {sim+1}/{n_simulations}")
        
        random.seed(sim)
        np.random.seed(sim)
        torch.manual_seed(sim)
        
        #start with initial pool
        labeled_indices = []
        unlabeled_indices = list(range(len(train_ds)))
        
        for size in dataset_sizes:
            print(f"Training on {size} samples...")
            
            #num new samples to select
            n_new_samples = size - len(labeled_indices)
            
            if n_new_samples > 0:
                unlabeled_features = features[unlabeled_indices]
                
                #select samples w/ density-diversity sampling
                new_indices_local = density_diversity_sampling(
                    unlabeled_features, 
                    n_new_samples, 
                    k=10, 
                    alpha=0.7
                )                
                new_indices_global = [unlabeled_indices[i] for i in new_indices_local]                
                labeled_indices.extend(new_indices_global)
                unlabeled_indices = [i for i in unlabeled_indices if i not in new_indices_global]
            
            #evaluate model on current labeled set
            train_dice, test_dice = evaluate_model_on_subset(train_ds, labeled_indices, test_loader)                        
            train_results.setdefault(size, []).append(train_dice)
            test_results.setdefault(size, []).append(test_dice)

    train_means = np.array([np.mean(train_results[s]) for s in dataset_sizes])
    train_std = np.array([np.std(train_results[s]) for s in dataset_sizes])
    test_means = np.array([np.mean(test_results[s]) for s in dataset_sizes])
    test_std = np.array([np.std(test_results[s]) for s in dataset_sizes])

    #plot
    plt.figure(figsize=(10, 6))
    plt.plot(dataset_sizes, train_means, '-o', label='Density-Based Active Learning')
    plt.fill_between(dataset_sizes, train_means - train_std, train_means + train_std, alpha=0.3)
    plt.title("Density-Based Active Learning: Mean Training Dice Score vs Training Set Size")
    plt.xlabel("Training Set Size")
    plt.ylabel("Mean Training Dice Score")
    plt.grid(True)
    plt.savefig("DensityActiveLearningTrainingDiceScore.png", bbox_inches='tight')
    plt.close()

    #plot
    plt.figure(figsize=(10, 6))
    plt.plot(dataset_sizes, test_means, '-o', label='Density-Based Active Learning')
    plt.fill_between(dataset_sizes, test_means - test_std, test_means + test_std, alpha=0.3)
    plt.title("Density-Based Active Learning: Mean Test Dice Score vs Training Set Size")
    plt.xlabel("Training Set Size")
    plt.ylabel("Mean Test Dice Score")
    plt.grid(True)
    plt.savefig("DensityActiveLearningTestDiceScore.png", bbox_inches='tight')
    plt.close()

    train_df = pd.DataFrame(train_results)
    train_df.to_csv("DensityActiveLearningTrainDiceScores.csv", index=False)
    test_df = pd.DataFrame(test_results)
    test_df.to_csv("DensityActiveLearningTestDiceScores.csv", index=False)
    print("Saved train/test Dice scores to CSV")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
        activation="sigmoid"
    ).to(device)

    final_subset = Subset(train_ds, labeled_indices)
    final_loader = DataLoader(final_subset, batch_size=8, shuffle=True)  # Increased batch size

    loss_fn = smp.losses.DiceLoss(mode='binary')
    optimizer = torch.optim.Adam(final_model.parameters(), lr=1e-4)

    for epoch in range(5):
        final_model.train()
        total_loss = 0.0
        for imgs, masks, _ in final_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = final_model(imgs)
            loss = loss_fn(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Final Model - Epoch {epoch+1}] Training Loss: {total_loss / len(final_loader):.4f}")

    torch.save(final_model.state_dict(), "resnet34_model_density_active_learning.pt")

if __name__ == "__main__":
    main()
