import wandb
import os
import time
from exam_project.model import BaseANN, BaseCNN, ViTClassifier
import torch


MODELS = {
    'ann': BaseANN,
    'cnn': BaseCNN,
    'vit': ViTClassifier
}

def load_model(artifact):

    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY_ORG")
                   , "project": os.getenv("WANDB_PROJECT")},
    )

    artifact = api.artifact(f"{os.getenv("MODEL_NAME")}",type="Model")
    artifact.download(root="./artifacts")
    file_name = artifact.files()[0].name
    model = MODELS[os.getenv("MODEL_ARCHITECTURE")]
    return model.load_from_checkpoint(f"./artifacts/{file_name}")

def test_model_speed():
    model = load_model(os.getenv("MODEL_NAME"))
    start = time.time()
    for _ in range(100):
        model(torch.rand(1, 1, 48, 48))
    end = time.time()
    assert end - start < 1



if __name__ == '__main__':
    test_model_speed()



