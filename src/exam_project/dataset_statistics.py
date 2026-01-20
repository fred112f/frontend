from exam_project.data import load_data

import torch
import hydra


@hydra.main(config_path="configs", config_name="train", version_base=None)
def dataset_statistics(cfg):
    train, val, test = load_data(processed_dir='data/processed/')
    train = torch.utils.data.DataLoader(train, persistent_workers=True, num_workers=9, batch_size=cfg.data.batch_size)
    val = torch.utils.data.DataLoader(val, persistent_workers=True, num_workers=9, batch_size=cfg.data.batch_size)
    test = torch.utils.data.DataLoader(test, persistent_workers=True, num_workers=9, batch_size=cfg.data.batch_size)

    print(train)
    print(val)
    print(test)



if __name__ == '__main__':
    dataset_statistics()