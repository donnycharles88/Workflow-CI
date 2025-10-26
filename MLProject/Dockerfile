FROM python:3.12-slim

WORKDIR /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies (versi stabil dan saling kompatibel)
RUN pip install --no-cache-dir \
    mlflow==2.19.0 \
    numpy==1.26.4 \
    pandas==2.2.3 \
    scikit-learn==1.6.0 \
    joblib \
    scipy==1.14.1 \
    dagshub==0.5.1 \
    seaborn==0.12.2 \
    matplotlib==3.8.0

# Default command: jalankan MLflow project
CMD ["mlflow", "run", ".", "--env-manager=local"]
