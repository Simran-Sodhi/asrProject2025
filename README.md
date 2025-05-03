# Leveraging Active Learning for Cell Identification in Phase Contrast Microscopic Images

This repository contains code and data for our project exploring active learning strategies—uncertainty sampling and density-diversity sampling—for efficient cell segmentation from phase-contrast microscopy images using the LIVECell dataset.

---

## Folder Structure

### `data_processing/`
Contains code to convert COCO-format annotations into PNG masks.

- `data_preprocessing.ipynb`: Converts COCO annotations into binary mask images using `pycocotools`.  
  **Data required**: Please download from the [LIVECell dataset website](https://sartorius-research.github.io/LIVECell/).

---

### `passive_learning/`
Implements the passive learning pipeline.

- `old/`: Contains early implementations using a naive U-Net and versions without partial training support.
- `local_passive_learning_pretrained_partial_training_option.py`: Main script to run the passive learning loop **locally** with pretrained U-Net and optional partial training.
- `aws_passive_learning_pretrained_partial_training_option.py`: Same functionality tailored for **AWS/EC2** instances.

---

### `uncertainty_sampling/`
Implements uncertainty-based active learning.

- `uncertainty_sampling_final_without_fisher.py`: Uses entropy-based uncertainty only.
- `uncertainty_sampling_final_with_fisher.py`: Adds Fisher Information to score sample impact and combine it with uncertainty.

---

### `density_sampling/`
Implements the density-diversity based sampling method.

- Contains a single script to run the active learning loop using density and diversity trade-offs.

---

### Root-Level Script
- `cell_type_predictions.ipynb`: Generates predictions for specific cell types using trained model checkpoints.  
  **Model checkpoints** are provided in the `results/` folder of the accompanying zip archive.

---

## Data Structure

Expected folder structure under `data/`:

```
data/
├── images_train/ # Training images (PNG)
├── images_mask/ # Training masks (PNG)
├── images_test/ # Test images (PNG)
├── images_test_masks/ # Test masks (PNG)
├── images_val/ # Validation images (PNG)
├── images_val_masks/ # Validation masks (PNG)
```


These folders are included in the zip archive provided with this submission.

---

## Running the Code

- To run **passive learning** locally:
  ```bash
  python local_passive_learning_pretrained_partial_training_option.py

- To run on AWS EC2:
    ```bash
    python aws_passive_learning_pretrained_partial_training_option.py

- To run uncertainty sampling:
     ```bash
    python uncertainty_sampling/uncertainty_sampling_final_without_fisher.py
    # or
    python uncertainty_sampling/uncertainty_sampling_final_with_fisher.py

- To run density sampling:
    ```bash
    python density_sampling/density_sampling.py

- Note: Except for local_passive_learning_pretrained_partial_training_option.py, most scripts assume access to boto3 for S3 uploading. If running locally without S3 setup, please comment out S3-related sections in the scripts.

### Citation
Dataset Source:
Edlund et al., 2021. LIVECell: A Large-scale Dataset for Label-free Live Cell Segmentation