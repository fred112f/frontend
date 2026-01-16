#!/bin/bash
set -e   # exit if any command fails

#Ensure GOOGLE_APPLICATION_CREDENTIALS is set
if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "ERROR: GOOGLE_APPLICATION_CREDENTIALS is not set!"
    exit 1
fi


# Step Inside App
cd /app

#Pull data
#uv run dvc pull -f gcsremote gs://dtu-mlops-exam-project-data
# Make sure the remote is set
uv run dvc remote add -f gcsremote gs://dtu-mlops-exam-project-data || true

# Pull the dataset
uv run dvc pull data.dvc -r gcsremote

# Run training
uv run src/exam_project/train_gcp.py

'''# Run this at the command line
docker run  --rm \
  --env-file $(pwd)/.env \
  -v $(pwd)/decent-seeker-484209-j2-8707d48bfe74.json:/app/key.json:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/key.json \
  --name trainingcpcontainer traingcp:latest
'''