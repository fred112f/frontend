FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

#Dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

#Set working dir
WORKDIR /app

#Copy Project
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src src/
COPY data.dvc data.dvc
COPY .dvc .dvc/
COPY .git .git/
COPY entrypoint.sh entrypoint.sh
#COPY models models/

#Set environment variables
ENV WANDB_ENTITY=krusand-danmarks-tekniske-universitet-dtu
ENV WANDB_PROJECT=MLOps-exam
ENV AIP_MODEL_DIR=models
ENV DATA_DIR=data/processed
ENV UV_LINK_MODE=copy

#Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv uv sync

#Executed when building image
RUN chmod +x entrypoint.sh

#Entrypoint (only executed when running container)
ENTRYPOINT ["/app/entrypoint.sh"]