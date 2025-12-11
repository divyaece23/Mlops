#!/bin/bash
# run_training.sh

# --- Configuration (can be imported from config.yaml or set here) ---
MLFLOW_TRACKING_URI="http://mlflow-tracking-server-xyz.a.run.app"
GCP_SA_KEY_PATH="/path/to/your/service-account-key.json"

# --- 1. Set Up the Environment ---

echo "1. Installing Python dependencies..."
pip install -r requirements.txt

echo "2. Authenticating to GCP for GCS access..."
# Use a service account key file for non-interactive environments
export GOOGLE_APPLICATION_CREDENTIALS="${GCP_SA_KEY_PATH}"

echo "3. Setting MLflow Environment Variables..."
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}"

# --- 2. Run the Training Script ---

echo "4. Executing model training and logging..."
python src/training/train.py

echo "Training and logging complete!"