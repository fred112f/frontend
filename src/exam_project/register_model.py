import wandb

run = wandb.init(entity="krusand-danmarks-tekniske-universitet-dtu-org", project="MLOps-exam")
artifact = run.use_artifact('krusand-danmarks-tekniske-universitet-dtu/MLOps-exam/model-brqq2gd7:v0', type='model')
artifact_dir = artifact.download()
run.link_artifact(
    artifact=artifact,
    target_path="model-registry/Fer-model",
    aliases=["latest"]
)