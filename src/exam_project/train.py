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

    run = wandb.init(
        project=cfg.logger.wandb.project,
        entity=cfg.logger.wandb.entity,
        job_type=cfg.logger.wandb.job_type,
        config=cfg_omega
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='validation_loss',
        dirpath='models/',
        filename='emotion-model-{epoch:02d}-{validation_loss:.2f}'
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
    train, val, test = load_data(processed_dir='data/processed/')
    train = torch.utils.data.DataLoader(train, persistent_workers=True, num_workers=9, batch_size=cfg.data.batch_size)
    val = torch.utils.data.DataLoader(val, persistent_workers=True, num_workers=9, batch_size=cfg.data.batch_size)
    test = torch.utils.data.DataLoader(test, persistent_workers=True, num_workers=9, batch_size=cfg.data.batch_size)
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
        name=f"emotion-model-{cfg.models._target_}",
        type="model",
        description="Emotion recognition model"
    )
    logger.info(artifact)
    # Add the model file to the artifact
    artifact.add_file(best_model_path)
    
    # Log the artifact
    wandb.log_artifact(artifact)
    logger.info("Artifact created and logged")
    logger.info("Linking artifact")
    # Link to model registry
    wandb.run.link_artifact(
        artifact=artifact,
        target_path="krusand-danmarks-tekniske-universitet-dtu-org/wandb-registry-fer-model/Model new",
        aliases=["latest"]
    )
    logger.info("Artifact linked")
    run.finish()
    logger.info("Training script finished")

if __name__ == "__main__":
    train()
