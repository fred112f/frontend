import os
import tempfile

from evidently.legacy.metrics import DataDriftTable
from evidently.legacy.report import Report
from google.cloud import storage
import hydra
from hydra.utils import get_original_cwd
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from evaluate import get_predictions
from model import BaseANN, BaseCNN, ViTClassifier


DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
MODELS = [("ann", BaseANN),
          ("cnn", BaseCNN),
          ("vit", ViTClassifier),
          ]

def extract_features(images):
    """Extract basic image features from a set of images."""
    # Convert PyTorch tensors to NumPy
    if isinstance(images, torch.Tensor):
        images = images.numpy()
    
    features = []
    for img in images:
        avg_brightness = np.mean(img)
        contrast = np.std(img)
        features.append([avg_brightness, contrast])
    return np.array(features)


@hydra.main(config_path="configs", config_name="data", version_base=None)
def main(cfg):
    # Tensor transformation for image datasets
    transform = transforms.Compose(
        [transforms.Resize((48, 48)),
         transforms.ToTensor(),]
         )

    # Paths
    original_cwd = get_original_cwd()
    input_dir = os.path.join(original_cwd, cfg.paths.data_root, "mma_sample", cfg.paths.raw_str)
    output_dir = os.path.join(original_cwd, cfg.paths.data_root, "mma_sample", cfg.paths.processed_str)
    fer_processed_dir = os.path.join(original_cwd, cfg.paths.data_root, cfg.paths.processed_str)
    os.makedirs(output_dir, exist_ok=True)

    mma_images = []
    mma_labels = []

    # Mapping class names to integer labels
    class_names = sorted(os.listdir(input_dir))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    # Looping through subfolders
    for cls_name in class_names:
        cls_folder = os.path.join(input_dir, cls_name)
        # if not os.path.isdir(cls_folder):
        #     continue
        for img_name in os.listdir(cls_folder):
            img_path = os.path.join(cls_folder, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                img = transform(img)
                mma_images.append(img)
                mma_labels.append(class_to_idx[cls_name])
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    # Stacking images and labels into tensors
    mma_image_tensor = torch.stack(mma_images)           # Shape: [N, C, H, W]
    mma_label_tensor = torch.tensor(mma_labels)          # Shape: [N]
    mma_test_set = torch.utils.data.TensorDataset(mma_image_tensor, mma_label_tensor)

    # Extracting features from mma_image_tensor
    mma_features = extract_features(mma_image_tensor)
    
    # Loading fer_image_tensor and labels and extracting features
    fer_image_tensor = torch.load(f"{fer_processed_dir}/test_images.pt")
    fer_label_tensor = torch.load(f"{fer_processed_dir}/test_target.pt")
    fer_test_set = torch.utils.data.TensorDataset(fer_image_tensor, fer_label_tensor)
    fer_images_np = fer_image_tensor.numpy()
    fer_features = extract_features(fer_images_np)

    # List of features
    feature_columns = ["Average Brightness", "Contrast"]

    # Constructing separate dataframes
    mma_df = np.column_stack((mma_features, ["mma"] * mma_features.shape[0]))
    fer_df = np.column_stack((fer_features, ["fer"] * fer_features.shape[0]))

    # Combining features
    combined_features = np.vstack((mma_df, fer_df))
    feature_df = pd.DataFrame(combined_features, columns=feature_columns + ["Dataset"])
    feature_df[feature_columns] = feature_df[feature_columns].astype(float)

    # Final dataframes for evidently
    reference_data = feature_df[feature_df["Dataset"] == "mma"].drop(columns=["Dataset"])
    current_data = feature_df[feature_df["Dataset"] == "fer"].drop(columns=["Dataset"])
 
    # Generating data drift report
    report = Report(metrics=[DataDriftTable()])
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html("reports/data_drift.html")

    # Initialize GCP storage client
    storage_client = storage.Client(project="decent-seeker-484209-j2")
    bucket = storage_client.bucket("dtu-mlops-exam-project-data")

    # Iterate over models
    for model_name, model_class in tqdm(MODELS):
        # Find model checkpoints in models/cnn folder
        blobs = list(bucket.list_blobs(prefix=f"models/{model_name}"))
        ckpt_files = [blob for blob in blobs if blob.name.endswith(".ckpt")]

        if not ckpt_files:
            raise FileNotFoundError("No .ckpt files found in the bucket!")

        # (optional) pick latest checkpoint
        ckpt_files.sort(key=lambda b: b.updated, reverse=True)
        ckpt_blob = ckpt_files[0]

        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
            ckpt_blob.download_to_filename(f.name)
            ckpt_path = f.name

        model = model_class.load_from_checkpoint(ckpt_path)
        model.eval()
        
        # Converting mma to grayscale for inference
        mma_image_tensor_gray = mma_image_tensor.mean(dim=1, keepdim=True)
        mma_test_set = torch.utils.data.TensorDataset(mma_image_tensor_gray, mma_label_tensor)

        # Dataloaders
        mma_test_loader = DataLoader(mma_test_set, persistent_workers=True, num_workers=9)
        fer_test_loader = DataLoader(fer_test_set, persistent_workers=True, num_workers=9)

        # Get predictions
        y_pred_mma, y_true_mma = get_predictions(model, mma_test_loader, DEVICE)
        y_pred_fer, y_true_fer = get_predictions(model, fer_test_loader, DEVICE)

        # Compute evaluation metrics
        mma_acc = accuracy_score(y_true_mma, y_pred_mma)
        mma_f1 = f1_score(y_true_mma, y_pred_mma, average="weighted")
        fer_acc = accuracy_score(y_true_fer, y_pred_fer)
        fer_f1 = f1_score(y_true_fer, y_pred_fer, average="weighted")

        # Comparing evaluation metrics
        print(f"Evaluating {model_name}: \n")
        print("\t On MMA (FER with colour)")
        print(f"\t\t Accuracy: {mma_acc:.4f}")
        print(f"\t\t F1-score: {mma_f1:.4f} \n")
        
        print("\t On FER")
        print(f"\t\t Accuracy: {fer_acc:.4f}")
        print(f"\t\t F1-score: {fer_f1:.4f} \n")


if __name__ == "__main__":
    main()


