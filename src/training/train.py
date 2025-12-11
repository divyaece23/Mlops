import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from google.cloud import storage # Required for GCS data storage (if used)

# --- GCP and MLflow Configuration ---
# You would typically set these as environment variables
GCP_PROJECT_ID = "YOUR_GCP_PROJECT_ID"
MLFLOW_TRACKING_URI = "http://your-mlflow-server:5000" # Replace with your MLflow tracking server URI
MLFLOW_EXPERIMENT_NAME = "Iris_Classification"
MODEL_NAME = "IrisLogisticRegressionModel"
GCS_DATA_BUCKET = "your-gcs-data-bucket-name" # Used if data is stored in GCS

def load_iris_data():
    """Loads Iris dataset from scikit-learn (simulates loading from GCS/DB)."""
    # For a real project, you would use google-cloud-storage to download a CSV
    # from GCS, or an equivalent for a database (e.g., BigQuery, Cloud SQL).
    # Example for GCS:
    # client = storage.Client(project=GCP_PROJECT_ID)
    # bucket = client.bucket(GCS_DATA_BUCKET)
    # blob = bucket.blob("iris/iris.csv")
    # blob.download_to_filename("iris.csv")
    
    iris = load_iris(as_frame=True)
    df = iris.frame
    X = df.drop(columns=['target'])
    y = df['target']
    return X, y

def train_and_log_model():
    X, y = load_iris_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set up MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        # Define and train model
        lr = LogisticRegression(solver='liblinear', multi_class='auto')
        lr.fit(X_train, y_train)
        predictions = lr.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Log parameters and metrics
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", accuracy)

        # Log the model (MLflow will handle dependencies and serialization)
        mlflow.sklearn.log_model(
            sk_model=lr, 
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )
        
        print(f"Model logged with run_id: {run.info.run_id}")
        print(f"Accuracy: {accuracy}")
        
    # --- For Production: Transition the model to 'Staging' or 'Production' ---
    # client = mlflow.tracking.MlflowClient()
    # client.transition_model_version_stage(
    #     name=MODEL_NAME,
    #     version=1, # The version you want to promote (check MLflow UI)
    #     stage="Staging" # or "Production"
    # )

if __name__ == "__main__":
    train_and_log_model()