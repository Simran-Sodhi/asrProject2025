# This script is also meant for AWS, if you want to run it on local you may comment out boto3 and s3 saving
# This script is identical to the newer scripts for passive learning except that this does not allow partial training
# Any functionality that this script provides is provided better through the newer script. To get the same effect set USE_WARM_START = False in the new script
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
import segmentation_models_pytorch as smp

import cv2
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import boto3

os.makedirs("results", exist_ok=True)

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

# subset_indices = random.sample(range(len(test_ds)), 10)
# test_subset = Subset(test_ds, subset_indices)
# test_loader =  DataLoader(test_subset, batch_size=1)

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

"""## Training"""

# loss_fn = smp.losses.DiceLoss(mode='binary')
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0.0
#     for imgs, masks, _ in train_loader:
#         imgs, masks = imgs.to(device), masks.to(device)

#         preds = model(imgs)
#         loss = loss_fn(preds, masks)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     print(f"[Epoch {epoch+1}] Training Loss: {total_loss / len(train_loader):.4f}")

"""## Evaluation/Testing"""

# model.eval()
# test_dice = 0.0
# with torch.no_grad():
#     for img, mask, _ in test_loader:
#         img, mask = img.to(device), mask.to(device)
#         pred = model(img)
#         pred_bin = (pred > 0.5).float()

#         intersection = (pred_bin * mask).sum()
#         union = pred_bin.sum() + mask.sum()
#         dice = (2 * intersection) / (union + 1e-8)
#         test_dice += dice.item()

# print(f"Test Dice Score: {test_dice / len(test_loader):.4f}")

"""## Visualization of test predictions"""


# def show_prediction(img, mask, filename, save=True):
#     model.eval()
#     with torch.no_grad():
#         pred = model(img.unsqueeze(0).to(device))
#         pred_bin = (pred > 0.5).float().squeeze().cpu().numpy()

#     pred_unpadded = unpad_to_shape(pred_bin, 520, 704)
#     img_unpadded = unpad_to_shape(img.squeeze(0), 520, 704)
#     mask_unpadded = unpad_to_shape(mask.squeeze(0), 520, 704)

#     fig, axs = plt.subplots(1, 3, figsize=(15, 5))

#     axs[0].imshow(img_unpadded, cmap='gray')
#     axs[0].set_title("Input Image")
#     axs[0].axis('off')

#     axs[1].imshow(mask_unpadded, cmap='gray')
#     axs[1].set_title("Ground Truth")
#     axs[1].axis('off')

#     axs[2].imshow(pred_unpadded, cmap='gray')
#     axs[2].set_title("Predicted Mask")
#     axs[2].axis('off')

#     plt.tight_layout()

#     if save:
#         # Ensure filename is safe
#         base_name = os.path.splitext(os.path.basename(filename))[0]
#         save_path = f"results/{base_name}_prediction.png"
#         plt.savefig(save_path, bbox_inches='tight')
#         print(f"Saved prediction to {save_path}")
#     else:
#         plt.show()
#     plt.close(fig)

# for test in test_ds:
#     img, mask, fname = test
#     show_prediction(img, mask, filename=fname)

"""# Passive Learning Style Training"""
# Function to allow passive learning
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

initial_size = 100
increment = 200
max_size = int(0.8 * len(train_ds))
n_simulations = 5

all_indices = list(range(len(train_ds)))
dataset_sizes = list(range(initial_size, max_size + 1, increment))

train_results, test_results = {}, {}
# Passive Learning Loop
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
means_train = np.array([np.mean(train_results[s]) for s in dataset_sizes])
stds_train = np.array([np.std(train_results[s]) for s in dataset_sizes])
plt.plot(dataset_sizes, means_train, '-o')
plt.fill_between(dataset_sizes, means_train - stds_train, means_train + stds_train, alpha=0.3)
#plt.errorbar(dataset_sizes, means_train, yerr=std_train, fmt='-o', capsize=5)
plt.title("Passive Learning: Mean Training Dice Score vs Training Set Size")
plt.xlabel("Training Set Size")
plt.ylabel("Mean Test Set Dice Score")
plt.grid(True)
plt.savefig("PassiveLearningMeanTrainingDiceScoreLighterRun.png", bbox_inches='tight')
print("Saved Figure")
plt.show()

means_test = np.array([np.mean(test_results[s]) for s in dataset_sizes])
stds_test = np.array([np.std(test_results[s]) for s in dataset_sizes])
plt.plot(dataset_sizes, means_test, '-o')
plt.fill_between(dataset_sizes, means_test - stds_test, means_test + stds_test, alpha=0.3)
#plt.errorbar(dataset_sizes, means_test, yerr=stds_test, fmt='-o', capsize=5)
plt.title("Passive Learning: Mean Test Set Dice Score vs Training Set Size")
plt.xlabel("Training Set Size")
plt.ylabel("Mean Test Set Dice Score")
plt.grid(True)
plt.savefig("PassiveLearningMeanTestDiceScoreLighterRun.png", bbox_inches='tight')
print("Saved Figure")
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(dataset_sizes, means_train, label='Train Dice (Mean)', color='blue', marker='o')
plt.plot(dataset_sizes, means_test, label='Test Dice (Mean)', color='orange', marker='o')
plt.fill_between(dataset_sizes, means_train - stds_train, means_train + stds_train, color='blue', alpha=0.3)
plt.fill_between(dataset_sizes, means_test - stds_test, means_test + stds_test, color='orange', alpha=0.3)

# Labels and legend
plt.title("Passive Learning: Mean Dice Score vs Training Set Size")
plt.xlabel("Training Set Size")
plt.ylabel("Mean Dice Score")
plt.legend()
plt.legend(loc="lower right", fontsize=12)
plt.grid(True)

# Save or show
plt.tight_layout()
plt.savefig("PassiveLearningMeanBothDiceScoreLighterRun.png", dpi=300)
plt.show()

train_df = pd.DataFrame(train_results)
train_df.to_csv("PassiveLearningTrainDiceScoresLighterRun.csv", index=False)

test_df = pd.DataFrame(test_results)
test_df.to_csv("PassiveLearningTestDiceScoresLighterRun.csv", index=False)

print("Saved train/test Dice scores to CSV")

#torch.save(model.state_dict(), "resnet34_model_all_data.pt")

BUCKET_NAME = 'live-cell-data'

# Initialize the boto3 S3 client
s3 = boto3.client('s3')

# Upload individual files
#s3.upload_file('resnet34_model_all_data.pt', BUCKET_NAME, 'resnet34_model_all_data.pt')
s3.upload_file('PassiveLearningTrainDiceScoresLighterRun.csv', BUCKET_NAME, 'PassiveLearningTrainDiceScoresLighterRun.csv')
s3.upload_file('PassiveLearningTestDiceScoresLighterRun.csv', BUCKET_NAME, 'PassiveLearningTestDiceScoresLighterRun.csv')
s3.upload_file('PassiveLearningMeanTestDiceScoreLighterRun.png', BUCKET_NAME, 'PassiveLearningMeanTestDiceScoreLighterRun.png')
s3.upload_file('PassiveLearningMeanTrainingDiceScoreLighterRun.png', BUCKET_NAME, 'PassiveLearningMeanTrainingDiceScoreLighterRun.png')

# Upload all files in the 'results/' folder
# results_dir = 'results'
# for filename in os.listdir(results_dir):
#     local_path = os.path.join(results_dir, filename)
#     s3_path = f"results/{filename}"  # You can customize this path in the bucket
#     if os.path.isfile(local_path):
#         print(f"Uploading {local_path} to s3://{BUCKET_NAME}/{s3_path}")
#         s3.upload_file(local_path, BUCKET_NAME, s3_path)


os.system('sudo shutdown now')