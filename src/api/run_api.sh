#!/bin/bash
# Start Gunicorn WSGI server to run the Flask app
# The '--bind :$PORT' is essential for Cloud Run, which assigns a port via the PORT env var.
gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 4 app:app