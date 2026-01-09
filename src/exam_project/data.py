# =========================
# Imports
# =========================

import kagglehub                # Download datasets from Kaggle
import os                       # File and path utilities
import PIL                      # Image processing
import torch                    # PyTorch core

from pathlib import Path        # OS-independent path handling
from PIL import Image           # PIL image loader

from torchvision import transforms
from torchvision.datasets import DatasetFolder

from typing import Callable, Dict, Literal


# =========================
# Configuration / Hyperparameters
# =========================

# Kaggle dataset identifiers
kaggle_id = 'msambare/fer2013'
data_version_path = '1'

# Preprocessing parameters
root = 'data'
raw_str = 'raw'
processed_str = 'processed'
trainvalsplit = 0.8
seed = 42


# =========================
# Type aliases (for clarity)
# =========================

# A transform maps a PIL image → torch tensor
Transform = Callable[[Image.Image], torch.Tensor]

# Allowed dataset split modes
TrainValTestMode = Literal["train", "val", "test"]


# =========================
# Directory utilities
# =========================

def create_processed_dir(processed_dir: str) -> None:
    """
    Create the data/processed directory if it does not exist.
    """
    Path(processed_dir).mkdir(parents=True, exist_ok=True)


# =========================
# Image loading & transforms
# =========================

def pil_loader(path: str) -> PIL.Image.Image:
    """
    Load an image from disk using PIL and convert it to grayscale.
    FER2013 images are single-channel (1 × H × W).
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')  # grayscale (1 channel)


def get_transform() -> Transform:
    """
    Create a preprocessing pipeline:
    - Convert PIL image to tensor in range [0, 1]
    - Normalize to range [-1, 1]
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    return transform


# =========================
# Dataset construction
# =========================

def get_dataset(root: str, transform: Transform) -> torch.utils.data.Dataset:
    """
    Load an image dataset from a directory using torchvision's DatasetFolder.
    Assumes images are stored in class-labeled subdirectories.
    """
    dataset = DatasetFolder(
        root=root,
        loader=pil_loader,
        extensions=['jpg'],
        transform=transform
    )
    return dataset


# =========================
# Tensor extraction helpers
# =========================

def get_image_labels_tensors(
    dataset: torch.utils.data.Dataset
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert an entire Dataset into stacked image and label tensors.
    """
    images_list = []
    labels_list = []

    for img, label in dataset:
        images_list.append(img)     # Tensor: [1, H, W]
        labels_list.append(label)   # Integer class index

    images_tensor = torch.stack(images_list)    # [N, 1, H, W]
    labels_tensor = torch.tensor(labels_list)   # [N]
    return images_tensor, labels_tensor


# =========================
# Saving utilities
# =========================

def save_image_labels(
    images: torch.Tensor,
    labels: torch.Tensor,
    processed_dir: str,
    traintest: TrainValTestMode
) -> None:
    """
    Save image and label tensors for a specific split.
    """
    torch.save(images, os.path.join(processed_dir, f"{traintest}_images.pt"))
    torch.save(labels, os.path.join(processed_dir, f"{traintest}_target.pt"))


def save_metadata(metadata: Dict[str, int], processed_dir: str) -> None:
    """
    Save class-to-index mapping (used for label decoding).
    """
    torch.save(metadata, os.path.join(processed_dir, 'class_to_idx.pt'))


# =========================
# Train/validation split
# =========================

def get_split_index(
    N: int,
    frac: float = 0.8
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate reproducible shuffled indices for train/validation splitting.
    """
    split = int(frac * N)
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(N, generator=g)

    train_idx = indices[:split]
    val_idx = indices[split:]
    return train_idx, val_idx


# =========================
# Dataset serialization
# =========================

def save_data(
    train_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    processed_dir: str,
    trainvalsplit: float
) -> None:
    """
    Save train, validation, and test datasets as .pt files.
    """
    # Convert full training dataset to tensors
    train_images_all, train_labels_all = get_image_labels_tensors(train_dataset)

    # Split into train and validation
    train_idx, val_idx = get_split_index(
        train_images_all.size(0),
        frac=trainvalsplit
    )

    # Save train & validation sets
    save_image_labels(
        train_images_all[train_idx],
        train_labels_all[train_idx],
        processed_dir,
        'train'
    )
    save_image_labels(
        train_images_all[val_idx],
        train_labels_all[val_idx],
        processed_dir,
        'val'
    )

    # Save test set
    save_image_labels(
        *get_image_labels_tensors(test_dataset),
        processed_dir,
        'test'
    )


# =========================
# End-to-end preprocessing
# =========================

def preprocess_data(
    raw_dir: str,
    processed_dir: str,
    trainvalsplit: float
) -> None:
    """
    Load raw image data, preprocess it, and save serialized tensors.
    """
    transform = get_transform()

    train_dataset = get_dataset(os.path.join(raw_dir, 'train'), transform)
    test_dataset = get_dataset(os.path.join(raw_dir, 'test'), transform)

    print('Converting datasets to .pt files...')
    save_data(train_dataset, test_dataset, processed_dir, trainvalsplit)
    save_metadata(train_dataset.class_to_idx, processed_dir)
    print('Done.')


# =========================
# Loading processed datasets
# =========================

def load_metadata(processed_dir: str) -> Dict[str, int]:
    """
    Load class-to-index metadata.
    """
    return torch.load(os.path.join(processed_dir, 'class_to_idx.pt'))


def load_data(
    processed_dir: str
) -> tuple[
    torch.utils.data.Dataset,
    torch.utils.data.Dataset,
    torch.utils.data.Dataset
]:
    """
    Load train, validation, and test TensorDatasets from disk.
    """
    train_images = torch.load(os.path.join(processed_dir, "train_images.pt"))
    train_target = torch.load(os.path.join(processed_dir, "train_target.pt"))

    val_images = torch.load(os.path.join(processed_dir, "val_images.pt"))
    val_target = torch.load(os.path.join(processed_dir, "val_target.pt"))

    test_images = torch.load(os.path.join(processed_dir, "test_images.pt"))
    test_target = torch.load(os.path.join(processed_dir, "test_target.pt"))

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    val_set = torch.utils.data.TensorDataset(val_images, val_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)

    return train_set, val_set, test_set


# =========================
# Script entry point
# =========================

if __name__ == "__main__":

    # Set KaggleHub cache directory
    os.environ["KAGGLEHUB_CACHE"] = os.path.join(root, raw_str)

    # Download dataset
    path = kagglehub.dataset_download(kaggle_id)
    print("Path to dataset files:", path)

    # Define directory paths
    raw_dir = os.path.join(
        root,
        f'{raw_str}/datasets/{kaggle_id}/versions/{data_version_path}/'
    )
    processed_dir = os.path.join(root, processed_str)

    # Create processed directory if needed
    create_processed_dir(processed_dir)

    # Sanity checks
    assert os.path.exists(raw_dir)
    assert os.path.exists(processed_dir)

    # Preprocess and save data
    preprocess_data(raw_dir, processed_dir, trainvalsplit)

    # Load processed datasets
    train_set, val_set, test_set = load_data(processed_dir)
