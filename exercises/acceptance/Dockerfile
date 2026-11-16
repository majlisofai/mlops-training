# The f1-points-finish API with access control (lesson 09) as a container image.
# The image holds the code, not the model and not the keys. The API loads MODEL_URI from the
# MLflow server at start-up and reads API_CLIENT_KEY and API_ADMIN_KEY from its environment.
#
# Build:  docker build -t lab09-api:1 exercises/acceptance
#         docker build -t lab09-api:ref --build-arg APP=solution/app.py exercises/acceptance
# Run:    docker run --rm --network host --env-file .secrets/api.env lab09-api:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
    MODEL_URI=models:/f1-points-finish@champion \
    PORT=8000

WORKDIR /app
# Install the packages first: Docker reuses this layer until requirements-serve.txt changes.
# Then remove pip: the API does not need it at run time, and the pip of the base image had
# known vulnerabilities when we audited it (audit_deps.py). What is not in the image cannot be attacked.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt && pip uninstall --yes pip

# Which app.py goes into the image: yours (default) or the reference solution.
ARG APP=app.py
COPY security.py .
COPY ${APP} app.py
# Do not run as root.
RUN useradd --create-home api
USER api

EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
