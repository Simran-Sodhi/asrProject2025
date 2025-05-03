# Uncertainty Sampling with No Batch Selection

""" 
This script implements a basic uncertainty sampling pipeline for active learning in image segmentation.
It uses a UNet model with a ResNet34 encoder to segment grayscale microscopy images into binary masks.
 
Key features:
- Uses pixel-wise entropy as the uncertainty measure to identify and query the most ambiguous samples.
- Simulates multiple active learning rounds by iteratively training on a growing labeled dataset and evaluating performance.
- Tracks and visualizes training and test Dice scores across iterations.
- Saves performance results as plots and CSVs, and uploads them to an AWS S3 bucket.
"""

import torch
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms
import os
import cv2
from PIL import Image
import pandas as pd
import heapq

import boto3

os.makedirs("results_UncertaintySamplingWithoutFisher", exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""## Data Class"""

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
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # (H, W)

        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Normalize
        image = image.astype('float32') / 255.0
        mask = (mask > 0).astype('float32')  # Binary mask

        # Convert to CHW format for PyTorch
        image = torch.tensor(image).unsqueeze(0)  # (1, H, W)
        mask = torch.tensor(mask).unsqueeze(0)    # (1, H, W)

        # Pad so that divisible by 32
        image = pad_to_multiple(image)
        mask = pad_to_multiple(mask)

        return image, mask, self.image_filenames[idx]

def pad_to_multiple(x, multiple=32):
    h, w = x.shape[-2], x.shape[-1]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return F.pad(x, (0, pad_w, 0, pad_h))

def unpad_to_shape(x, original_h, original_w):
    return x[..., :original_h, :original_w]

"""## Load Data"""

data_dir = "../data"
train_ds = CellSegmentationDataset(f"{data_dir}/images_train", f"{data_dir}/masks_train")
val_ds =  CellSegmentationDataset(f"{data_dir}/images_val", f"{data_dir}/masks_val")
test_ds = CellSegmentationDataset(f"{data_dir}/images_test", f"{data_dir}/masks_test")

# Do this on full dataset
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)
test_loader = DataLoader(test_ds, batch_size=1)

"""## UNet Model Definition"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1,
    activation="sigmoid"
).to(device)

"""# Uncertainty Sampling (based on entropy)"""

# Function to evaluate train and test DICE scores on a subset of the dataset
def evaluate_model_on_subset(dataset, subset_indices, test_loader, epochs=5):
    subset = Subset(dataset, subset_indices)
    loader = DataLoader(subset, batch_size=4, shuffle=True)

    model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=1, classes=1, activation="sigmoid").to(device)
    loss_fn = smp.losses.DiceLoss(mode='binary')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training
    model.train()
    for _ in range(epochs):
        for imgs, masks, _ in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # Evaluation on training set after last epoch
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
    final_train_dice = np.mean(train_dice_scores)

    # Evaluation on test set
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
    final_test_dice = np.mean(test_dice_scores)
    return final_train_dice, final_test_dice

"""# Uncertainty function:"""
# Function to compute uncertainty scores based on pixel-wise entropy
def get_uncertainty_scores(model, dataset, unlabeled_indices):
    model.eval()
    uncertainties = []
    epsilon = 1e-10  # Small value to avoid log(0)

    with torch.no_grad():
        for idx in unlabeled_indices:
            img, _, _ = dataset[idx]
            img = img.unsqueeze(0).to(device)

            # Predict probabilities using sigmoid
            pred = model(img).sigmoid()  # Shape: (1, 1, H, W)

            # Remove batch and channel dimensions -> shape: (H, W)
            probs = pred.squeeze().detach().cpu().numpy()

            # Compute pixel-wise entropy
            pixel_entropy = -probs * np.log(probs + epsilon) - (1 - probs) * np.log(1 - probs + epsilon)

            # Aggregate entropy (mean over all pixels)
            uncertainty = np.mean(pixel_entropy)

            uncertainties.append((uncertainty, idx))

    return uncertainties

"""# Uncertainty Sampling Simulation"""
# Run 5 simulations of uncertainty sampling
initial_size = 100
query_size = 200
max_size = int(0.5 * len(train_ds))
n_simulations = 5

train_results, test_results = {}, {}

for sim in range(n_simulations):
    random.seed(sim)
    all_indices = list(range(len(train_ds)))
    random.shuffle(all_indices)
    labeled_indices = all_indices[:initial_size]
    unlabeled_indices = all_indices[initial_size:]

    while len(labeled_indices) <= max_size:
        print(f"Training on {len(labeled_indices)} samples...")

        train_dice, test_dice = evaluate_model_on_subset(train_ds, labeled_indices, test_loader)
        print(f" Train Dice = {train_dice:.4f}", f" Test Dice = {test_dice:.4f}")

        train_results.setdefault(len(labeled_indices), []).append(train_dice)
        test_results.setdefault(len(labeled_indices), []).append(test_dice)

        # Stop if max size is reached
        if len(labeled_indices) + query_size > max_size:
            break

        # Train a model on the current labeled set
        model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=1, classes=1, activation="sigmoid").to(device)
        loader = DataLoader(Subset(train_ds, labeled_indices), batch_size=4, shuffle=True)
        loss_fn = smp.losses.DiceLoss(mode='binary')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        model.train()
        for epoch in range(3):  # few epochs just for uncertainty estimation
            for imgs, masks, _ in loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                loss = loss_fn(preds, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Select most uncertain samples
        scores = get_uncertainty_scores(model, train_ds, unlabeled_indices)
        scores.sort()  # Sort by uncertainty (closest to 0.5)
        selected = [idx for _, idx in scores[:query_size]]

        labeled_indices += selected
        unlabeled_indices = list(set(unlabeled_indices) - set(selected))

# Function to plot train/test dice scores
def plot_train_test(train_results, test_results, dataset_sizes, title, filename_prefix, color_train, color_test):
    train_means = np.array([np.mean(train_results[s]) for s in dataset_sizes])
    train_stds = np.array([np.std(train_results[s]) for s in dataset_sizes])
    test_means = np.array([np.mean(test_results[s]) for s in dataset_sizes])
    test_stds = np.array([np.std(test_results[s]) for s in dataset_sizes])

    plt.figure(figsize=(8, 6))
    plt.plot(dataset_sizes, train_means, '-o', label='Train Dice', color=color_train)
    plt.plot(dataset_sizes, test_means, '-o', label='Test Dice', color=color_test)
    plt.fill_between(dataset_sizes, train_means - train_stds, train_means + train_stds, alpha=0.3, color=color_train)
    plt.fill_between(dataset_sizes, test_means - test_stds, test_means + test_stds, alpha=0.3, color=color_test)
    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel("Mean Dice Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f"{filename_prefix}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")
    return filename

# Dataset sizes (must match keys in results dicts)
dataset_sizes = sorted(train_results.keys())

# Generate and Save Plot 
file1 = plot_train_test(train_results, test_results, dataset_sizes,
                        "Uncertainty Sampling: Dice Scores vs Training Set Size",
                        "UncertaintySampling_Without_Fisher_DiceScores", color_train="blue", color_test="orange")

# Save CSVs
csv_files = []
def save_df(data, name):
    df = pd.DataFrame(data)
    filename = f"{name}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {filename}")
    csv_files.append(filename)
    return filename

save_df(train_results, "UncertaintySampling_Without_Fisher_TrainDiceScores")
save_df(test_results, "UncertaintySampling_Without_Fisher_TestDiceScores")

print("Saved train/test Dice scores to CSV")

# Upload all plots and CSVs
BUCKET_NAME = 'asr25data'
s3 = boto3.client('s3')

all_files = [file1] + csv_files
for fname in all_files:
    s3.upload_file(fname, BUCKET_NAME, f"results/{fname}")
    print(f"Uploaded {fname} to s3://{BUCKET_NAME}/{fname}")

os.system('sudo shutdown now')
