from exam_project.data import load_data

import torch
import hydra
import matplotlib.pyplot as plt
import os

@hydra.main(config_path="configs", config_name="train", version_base=None)
def dataset_statistics(cfg):
    # Print statements are printed to the report, which is commented on PR
    train, val, test = load_data(processed_dir='data/processed/')
    train = torch.utils.data.DataLoader(train, persistent_workers=True, num_workers=9, batch_size=cfg.data.batch_size)
    val = torch.utils.data.DataLoader(val, persistent_workers=True, num_workers=9, batch_size=cfg.data.batch_size)
    test = torch.utils.data.DataLoader(test, persistent_workers=True, num_workers=9, batch_size=cfg.data.batch_size)

    print("Face emotion dataset")
    print(f"Number of training images: {len(train.dataset)}")
    print(f"Image shape/size: {train.dataset[0][0].shape}")
    print("\n")

    print(f"Number of validation images: {len(val.dataset)}")
    print(f"Image shape/size: {val.dataset[0][0].shape}")
    print("\n")
    

    print(f"Number of testing images: {len(test.dataset)}")
    print(f"Image shape/size: {test.dataset[0][0].shape}")
    print("\n")

    train_label_distribution = torch.bincount(train.dataset[:][1])
    val_label_distribution = torch.bincount(val.dataset[:][1])
    test_label_distribution = torch.bincount(test.dataset[:][1])

    class2idx = torch.load(os.path.join('data/processed/',"class_to_idx.pt"))

    plt.bar(class2idx.keys(), train_label_distribution)
    plt.title("Train label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("reports/figures/train_label_distribution.png")
    plt.close()

    plt.bar(class2idx.keys(), val_label_distribution)
    plt.title("Val label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("reports/figures/val_label_distribution.png")
    plt.close()

    plt.bar(class2idx.keys(), test_label_distribution)
    plt.title("Test label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("reports/figures/test_label_distribution.png")
    plt.close()


if __name__ == '__main__':
    dataset_statistics()