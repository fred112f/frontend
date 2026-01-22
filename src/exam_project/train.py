from exam_project.data import load_data

from loguru import logger
import hydra
from hydra.utils import instantiate
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import wandb
from omegaconf import OmegaConf
import os

from google.cloud import storage
import pytorch_lightning

DATA_DIR = os.environ.get("DATA_DIR", "data/processed/")
MODEL_DIR = os.environ.get("AIP_MODEL_DIR", "models")

NUM_WORKERS = int(os.getenv("NUM_WORKERS",f"{min(4, os.cpu_count())}"))
PERSISTENT_WORKERS = NUM_WORKERS>0

#Make model dir if it doesn't not already exist
os.makedirs(MODEL_DIR, exist_ok=True)

#Set random seed
pytorch_lightning.seed_everything(42, workers=True)

@hydra.main(config_path="configs", config_name="train", version_base=None)
def train(cfg):
    """
    Trains the model

    params: 
        cfg: .yaml using Hydra
    """

    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # Add a log file to the logger
    logger.remove()
    logger.add(os.path.join(hydra_path, "loguru_logging.log"), level=cfg.debug.level)
    logger.info("Training script started")
    logger.debug(cfg)
    cfg_omega = OmegaConf.to_container(cfg)
    model_name = hydra.core.hydra_config.HydraConfig.get().runtime.choices.models

    run = wandb.init(
        project=cfg.logger.wandb.project,
        entity=cfg.logger.wandb.entity,
        job_type=cfg.logger.wandb.job_type,
        config=cfg_omega
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='validation_loss',
        mode='min',#i.e. we are aiming for the minimum validation loss
        dirpath=MODEL_DIR,
        filename='emotion-model-{epoch:02d}-{validation_loss:.2f}',
        save_top_k=1
    )

    trainer_args = {"max_epochs": cfg.trainer.max_epochs
                    , 'accelerator': cfg.trainer.accelerator
                    , 'logger': WandbLogger(log_model=cfg.logger.wandb.log_model, project=cfg.logger.wandb.project)
                    , 'limit_train_batches': cfg.trainer.limit_train_batches
                    , 'limit_val_batches': cfg.trainer.limit_val_batches
                    , 'log_every_n_steps': cfg.trainer.log_every_n_steps
                    , "callbacks": [checkpoint_callback]}
    logger.debug(f"{trainer_args = }")
    logger.info("Finished cfg setup")
    logger.info("Starting dataloading")
    train, val, test = load_data(processed_dir=DATA_DIR)
    train = torch.utils.data.DataLoader(train, persistent_workers=PERSISTENT_WORKERS, num_workers=NUM_WORKERS, batch_size=cfg.data.batch_size)
    val = torch.utils.data.DataLoader(val, persistent_workers=PERSISTENT_WORKERS, num_workers=NUM_WORKERS, batch_size=cfg.data.batch_size)
    test = torch.utils.data.DataLoader(test, persistent_workers=PERSISTENT_WORKERS, num_workers=NUM_WORKERS, batch_size=cfg.data.batch_size)
    logger.info("Finished dataloading")

    logger.info("Loading model")
    model = instantiate(cfg.models)
    logger.info("Model loaded")
    trainer = Trainer(**trainer_args)
    logger.info("Model fitting started")
    trainer.fit(model=model, train_dataloaders=train, val_dataloaders=val)
    logger.info("Model fitting finished")
    # Save and log the best model to model registry
    best_model_path = checkpoint_callback.best_model_path
    logger.info(f"{best_model_path = }")

    logger.info("Creating artifact")
    # Create an artifact
    artifact = wandb.Artifact(
        name=f"emotion-model-{model_name}",
        type="Model",
        description="Emotion recognition model",
        metadata={'architecture': model_name, 'device': cfg.trainer.accelerator}
    )
    logger.info(artifact)
    # Add the model file to the artifact
    if best_model_path.startswith("gs://"): #W&B cannot add unless file is local
        local_model_path = "/tmp/" + os.path.basename(best_model_path)  # Temp local path
        logger.info(f"{local_model_path = }")
        
        # Download from GCS
        client = storage.Client()
        bucket_name, blob_path = best_model_path[5:].split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.download_to_filename(local_model_path)
        artifact.add_file(local_model_path)  # update to local path
    else:
        artifact.add_file(best_model_path)
    
    # Log the artifact
    wandb.log_artifact(artifact)
    logger.info("Artifact created and logged")
    logger.info("Linking artifact")

    # Link to model registry
    target_path = f"krusand-danmarks-tekniske-universitet-dtu-org/wandb-registry-fer-model/{model_name}"
    wandb.run.link_artifact(
        artifact=artifact,
        target_path=target_path,
        aliases=["latest", "staging"]
    )
    logger.info(target_path)
    logger.info("Artifact linked")
    run.finish()
    logger.info("Training script finished")

if __name__ == "__main__":
    train()
