# Use a lightweight Python base image
FROM python:3.9-slim

# Set environment variables (optional, but good practice)
ENV MODEL_NAME="IrisLogisticRegressionModel"
ENV MODEL_STAGE="Production"
# The MLFLOW_TRACKING_URI will be injected by Cloud Run/CI/CD secret manager for security.
# Example: ENV MLFLOW_TRACKING_URI="http://your-mlflow-server:5000"

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Make the startup script executable
RUN chmod +x run_api.sh

# Expose the port (Cloud Run manages external access)
EXPOSE 8080

# Command to run the production server
# Cloud Run will look for this command to start the container
CMD ["/app/run_api.sh"]