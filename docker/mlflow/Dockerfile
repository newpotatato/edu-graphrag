# syntax=docker/dockerfile:1

# MLflow Tracking Server. The official image ships without database and S3 drivers,
# so add psycopg2 (PostgreSQL backend store) and boto3 (S3 artifact store).
FROM ghcr.io/mlflow/mlflow:v3.17.1

RUN pip install --no-cache-dir \
        psycopg2-binary==2.9.13 \
        boto3==1.43.114

# The base image runs as root; the server needs no privileges.
RUN groupadd --gid 10001 mlflow \
    && useradd --uid 10001 --gid mlflow --create-home --shell /usr/sbin/nologin mlflow

USER 10001:10001
WORKDIR /home/mlflow

EXPOSE 5000

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=2)"]

# Store URIs and credentials are passed by docker-compose.yml.
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
