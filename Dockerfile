FROM apache/airflow:3.1.7

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy requirements file into container
COPY requirements.txt /tmp/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --user -r /tmp/requirements.txt