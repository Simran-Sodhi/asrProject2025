# This script is also meant for AWS, if you want to run it on local you may comment out boto3 and s3 saving
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset

import cv2
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import boto3

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


# Define naive from scratach model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2)
                )

        # Encoder levels
        self.enc1 = CBR(1, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)
        self.pool = nn.MaxPool2d(2)

        # Decoder levels
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = CBR(512, 256)  # using skip from enc3
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)  # using skip from enc2
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)   # using skip from enc1

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)                   # (B, 64, H, W)
        e2 = self.enc2(self.pool(e1))         # (B, 128, H/2, W/2)
        e3 = self.enc3(self.pool(e2))         # (B, 256, H/4, W/4)
        e4 = self.enc4(self.pool(e3))         # (B, 512, H/8, W/8)

        # Decoder
        d3 = self.up3(e4)                   # (B, 256, H/4, W/4)
        d3 = torch.cat([d3, e3], dim=1)       # (B, 256+256, H/4, W/4)
        d3 = self.dec3(d3)                  # (B, 256, H/4, W/4)

        d2 = self.up2(d3)                   # (B, 128, H/2, W/2)
        d2 = torch.cat([d2, e2], dim=1)       # (B, 128+128, H/2, W/2)
        d2 = self.dec2(d2)                  # (B, 128, H/2, W/2)

        d1 = self.up1(d2)                   # (B, 64, H, W)
        d1 = torch.cat([d1, e1], dim=1)       # (B, 64+64, H, W)
        d1 = self.dec1(d1)                  # (B, 64, H, W)

        out = self.final(d1)
        return torch.sigmoid(out)

    
"""## Load Data"""
# Adjust these file paths as required
data_dir = "../data"
train_ds = CellSegmentationDataset(f"{data_dir}/images_train", f"{data_dir}/masks_train")
val_ds =  CellSegmentationDataset(f"{data_dir}/images_val", f"{data_dir}/masks_val")
test_ds = CellSegmentationDataset(f"{data_dir}/images_test", f"{data_dir}/masks_test")

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)
test_loader = DataLoader(test_ds, batch_size=1)

"""## This is done cause of compute issues"""

# Randomly pick 10 indices from the training dataset
# subset_indices = random.sample(range(len(train_ds)), 10)

# Create a subset dataset
# train_subset = Subset(train_ds, subset_indices)

# train_loader = DataLoader(train_subset, batch_size=2, shuffle=True)

subset_indices = random.sample(range(len(test_ds)), 20)
test_subset = Subset(test_ds, subset_indices)
test_loader =  DataLoader(test_subset, batch_size=1)

"""## UNet Model Definition"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()

"""## Training"""

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for imgs, masks, _ in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)

        preds = model(imgs)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Epoch {epoch+1}] Training Loss: {total_loss / len(train_loader):.4f}")

"""## Evaluation/Testing"""

model.eval()
test_dice = 0.0
with torch.no_grad():
    for img, mask, _ in test_loader:
        img, mask = img.to(device), mask.to(device)
        pred = model(img)
        pred_bin = (pred > 0.5).float()

        intersection = (pred_bin * mask).sum()
        union = pred_bin.sum() + mask.sum()
        dice = (2 * intersection) / (union + 1e-8)
        test_dice += dice.item()

print(f"Test Dice Score: {test_dice / len(test_loader):.4f}")

"""## Visualization of test predictions"""

os.makedirs("results_naive_model", exist_ok=True)

# Helper Function to show predictions generated by the model along with ground truth mask and actual image
def show_prediction(img, mask, filename, save=True):
    # Prediction mode of model
    model.eval()
    # No learning
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device))
        pred_bin = (pred > 0.5).float().squeeze().cpu().numpy()
    # Original dimensions for all to view
    pred_unpadded = unpad_to_shape(pred_bin, 520, 704)
    img_unpadded = unpad_to_shape(img.squeeze(0), 520, 704)
    mask_unpadded = unpad_to_shape(mask.squeeze(0), 520, 704)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(img_unpadded, cmap='gray')
    axs[0].set_title("Input Image")
    axs[0].axis('off')

    axs[1].imshow(mask_unpadded, cmap='gray')
    axs[1].set_title("Ground Truth")
    axs[1].axis('off')

    axs[2].imshow(pred_unpadded, cmap='gray')
    axs[2].set_title("Predicted Mask")
    axs[2].axis('off')

    plt.tight_layout()

    if save:
        # Ensure filename is safe
        base_name = os.path.splitext(os.path.basename(filename))[0]
        save_path = f"results_naive_model/{base_name}_prediction.png"
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved prediction to {save_path}")
    else:
        plt.show()
    plt.close(fig)

for test in test_subset:
    img, mask, fname = test
    show_prediction(img, mask, filename=fname)

"""# Passive Learning Style Training"""
# Function to enable passive learning cycle
def evaluate_model_on_subset(dataset, subset_indices, test_loader, epochs=5):
    subset = Subset(dataset, subset_indices)
    loader = DataLoader(subset, batch_size=4, shuffle=True)
    # Instantiate model
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    # Training
    model.train()
    for _ in range(epochs):
        for imgs, masks, _ in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)

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

# Training Loop parameters
initial_size = 100
increment = 200
max_size = int(0.8 * len(train_ds))
n_simulations = 5

all_indices = list(range(len(train_ds)))
dataset_sizes = list(range(initial_size, max_size + 1, increment))

train_results, test_results = {}, {}
for sim in range(n_simulations):
    random.seed(sim)
    shuffled_indices = all_indices.copy()
    random.shuffle(shuffled_indices)
    for size in dataset_sizes:
        current_subset = shuffled_indices[:size]  # always includes previous ones
        print(f"  Training on {size} samples...", end="")
        train_dice, test_dice = evaluate_model_on_subset(train_ds, current_subset, test_loader)
        print(f" Train Dice = {train_dice:.4f}", f" Test Dice = {test_dice:.4f}")
        train_results.setdefault(size, []).append(train_dice)
        test_results.setdefault(size, []).append(test_dice)

# Plotting
means = np.array([np.mean(train_results[s]) for s in dataset_sizes])
std_dev = np.array([np.std(train_results[s]) for s in dataset_sizes])
plt.plot(dataset_sizes, means, '-o')
plt.fill_between(dataset_sizes, means - std_dev, means + std_dev, alpha=0.3)
#plt.errorbar(dataset_sizes, means, yerr=std_dev, fmt='-o', capsize=5)
plt.title("Passive Learning (Naive UNet Model): Mean Training Dice Score vs Training Set Size")
plt.xlabel("Training Set Size")
plt.ylabel("Mean Test Set Dice Score")
plt.grid(True)
plt.savefig("PassiveLearningMeanTrainingDiceScoreNaiveModel.png", bbox_inches='tight')
print("Saved Figure")
plt.show()

means = np.array([np.mean(test_results[s]) for s in dataset_sizes])
std_dev = np.array([np.std(test_results[s]) for s in dataset_sizes])
plt.plot(dataset_sizes, means, '-o')
plt.fill_between(dataset_sizes, means - std_dev, means + std_dev, alpha=0.3)
#plt.errorbar(dataset_sizes, means, yerr=std_dev, fmt='-o', capsize=5)
plt.title("Passive Learning (Naive UNet Model): Mean Test Set Dice Score vs Training Set Size")
plt.xlabel("Training Set Size")
plt.ylabel("Mean Test Set Dice Score")
plt.grid(True)
plt.savefig("PassiveLearningMeanTestDiceScoreNaiveModel.png", bbox_inches='tight')
print("Saved Figure")
plt.show()

train_df = pd.DataFrame(train_results)
train_df.to_csv("PassiveLearningTrainDiceScoresNaiveModel.csv", index=False)

test_df = pd.DataFrame(test_results)
test_df.to_csv("PassiveLearningTestDiceScoresNaiveModel.csv", index=False)

print("Saved train/test Dice scores to CSV")

torch.save(model.state_dict(), "naive_model_all_data.pt")

BUCKET_NAME = 'live-cell-data'

# Initialize the boto3 S3 client
s3 = boto3.client('s3')

# Upload individual files
s3.upload_file('naive_model_all_data.pt', BUCKET_NAME, 'naive_model_all_data.pt')
s3.upload_file('PassiveLearningTrainDiceScoresNaiveModel.csv', BUCKET_NAME, 'PassiveLearningTrainDiceScoresNaiveModel.csv')
s3.upload_file('PassiveLearningTestDiceScoresNaiveModel.csv', BUCKET_NAME, 'PassiveLearningTestDiceScoresNaiveModel.csv')
s3.upload_file('PassiveLearningMeanTestDiceScoreNaiveModel.png', BUCKET_NAME, 'PassiveLearningMeanTestDiceScoreNaiveModel.png')
s3.upload_file('PassiveLearningMeanTrainingDiceScoreNaiveModel.png', BUCKET_NAME, 'PassiveLearningMeanTrainingDiceScoreNaiveModel.png')

#Upload all files in the 'results/' folder
results_dir = 'results_naive_model'
for filename in os.listdir(results_dir):
    local_path = os.path.join(results_dir, filename)
    s3_path = f"results_naive_model/{filename}"
    if os.path.isfile(local_path):
        print(f"Uploading {local_path} to s3://{BUCKET_NAME}/{s3_path}")
        s3.upload_file(local_path, BUCKET_NAME, s3_path)


os.system('sudo shutdown now')