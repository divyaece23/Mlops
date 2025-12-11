from flask import Flask, request, jsonify
import mlflow.pyfunc
import os
import pandas as pd

# --- Configuration ---
# Cloud Run will inject the MLFLOW_TRACKING_URI from its environment
# The MLFLOW_TRACKING_URI must point to your MLflow Tracking Server (e.g., hosted on GCP App Engine or VM)
MODEL_NAME = os.getenv("MODEL_NAME", "IrisLogisticRegressionModel")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production") # Use 'Production' or 'Staging'

# Set MLflow tracking URI from environment for model loading
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = Flask(__name__)
model = None

def load_model():
    """Loads the latest model from the MLflow Model Registry."""
    global model
    try:
        # MLflow handles fetching the model artifact from the GCS/S3 location
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Successfully loaded model: {model_uri}")
    except Exception as e:
        print(f"Error loading model from MLflow: {e}")
        # In a real scenario, you would health-check this on startup
        raise RuntimeError("Failed to load ML model.") from e

# Load model on startup
load_model()

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for making predictions."""
    if not model:
        return jsonify({"error": "Model not loaded"}), 503
        
    try:
        # Expecting JSON input: {"data": [[5.1, 3.5, 1.4, 0.2]]}
        json_data = request.get_json(force=True)
        data = json_data.get('data')
        
        # Convert list of lists to Pandas DataFrame (MLflow models expect this)
        # Columns must match the training data feature order: sepal_length, sepal_width, petal_length, petal_width
        features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        input_df = pd.DataFrame(data, columns=features)
        
        # Make prediction
        predictions = model.predict(input_df)
        
        # Return predictions as a list
        return jsonify({
            "predictions": predictions.tolist()
        })
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 400

# The API is run using Gunicorn via run_api.sh, so we don't need app.run() here.