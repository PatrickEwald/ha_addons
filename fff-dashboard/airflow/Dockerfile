FROM apache/airflow:2.8.1

# Install system dependencies for ML packages
USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Copy requirements and install Python packages
COPY requirements.txt /opt/airflow/requirements.txt
RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt

# Verify installations
RUN python -c "import numpy; print(f'NumPy {numpy.__version__} installed')"
RUN python -c "import holidays; print(f'Holidays {holidays.__version__} installed')"
RUN python -c "import lightgbm; print(f'LightGBM {lightgbm.__version__} installed')"
RUN python -c "import sklearn; print(f'Scikit-learn {sklearn.__version__} installed')"

# Set working directory
WORKDIR /opt/airflow

# Health check
HEALTHCHECK CMD ["python", "-c", "import airflow; print('Airflow packages available')"]
