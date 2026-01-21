#!/bin/bash
set -e   # exit if any command fails

#Check for GOOGLE_APPLICATION_CREDENTIALS is set
if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "GOOGLE_APPLICATION_CREDENTIALS not set. Assuming running on GCP with default credentials."
else
    echo "Using GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS"
fi


# Step Inside App
cd /app

#Pull data
#uv run dvc pull -f gcsremote gs://dtu-mlops-exam-project-data
# Make sure the remote is set
echo "About to uv run dvc remote add..."
uv run dvc remote add -f gcsremote gs://dtu-mlops-exam-project-data || true

# Pull the dataset
echo "About to uv run dvc pull..."
uv run dvc pull data.dvc -r gcsremote --verbose

# Run training
echo "About to uv run train script..."
uv run src/exam_project/train_gcp.py

# Run this at the command line
: <<'COMMENT'
###
Local image local run
------------
docker build -f dockerfiles/train_gcp.dockerfile . -t traingcp:latest

docker run  --rm \
  --env-file $(pwd)/.env \
  -v $(pwd)/decent-seeker-484209-j2-8707d48bfe74.json:/app/key.json:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/key.json \
  --name trainingcpcontainer traingcp:latest

Remote image local run
------------
Push docker image to artifact registry, run it locally OR run it via Vertex AI
docker build -f dockerfiles/train_gcp.dockerfile -t europe-west1-docker.pkg.dev/decent-seeker-484209-j2/myartifactregistry/traingcp:latest --platform linux/amd64 --push .

#Mount local folder models/ to app/models and run
docker run  --rm \
  --env-file $(pwd)/.env \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/decent-seeker-484209-j2-8707d48bfe74.json:/app/key.json:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/key.json \
  --name trainingcpcontainer europe-west1-docker.pkg.dev/decent-seeker-484209-j2/myartifactregistry/traingcp:latest

Remote image remote run
------------
1) Create a service account and add storage.objectAdmin role
gcloud iam service-accounts create my-training-job-sa \
  --display-name="Training Job Service Account for DTU MLOps"

gcloud projects add-iam-policy-binding decent-seeker-484209-j2 \
  --member="serviceAccount:my-training-job-sa@decent-seeker-484209-j2.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

2) Configure the run job (without executing)
gcloud run jobs create my-training-job \
  --image=europe-west1-docker.pkg.dev/decent-seeker-484209-j2/myartifactregistry/traingcp:latest \
  --region=europe-west1 \
  --service-account=my-training-job-sa@decent-seeker-484209-j2.iam.gserviceaccount.com \
  --set-env-vars="AIP_MODEL_DIR=gs://dtu-mlops-exam-project-data/models,WANDB_API_KEY=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,WANDB_ENTITY=krusand-danmarks-tekniske-universitet-dtu,WANDB_PROJECT=MLOps-exam" \
  --memory=16Gi \
  --cpu=4 \
  --max-retries=0

3) Execute job
gcloud run jobs execute my-training-job \
  --region=europe-west1

Using Vertex AI
------------
0) Enable Vertex AI
gcloud services enable aiplatform.googleapis.com

1) Create service account and add roles:

gcloud iam service-accounts create vertexai-training-sa \
  --display-name="VertexAI Training Job Service Account for DTU MLOps"

gcloud projects add-iam-policy-binding decent-seeker-484209-j2 \
  --member="serviceAccount:vertexai-training-sa@decent-seeker-484209-j2.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding decent-seeker-484209-j2 \
  --member="serviceAccount:vertexai-training-sa@decent-seeker-484209-j2.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

2) Create Vertex AI custom job (Note Creating the Job IS Running the Job)

gcloud ai custom-jobs create \
  --region=europe-west1 \
  --display-name=vertexai-training-job \
  --config=vertex_ai_job.yaml 
COMMENT