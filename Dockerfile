FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY params.yaml .
COPY data/raw/ data/raw/

# Train inside the image so MLflow's registry records container-native
# artifact paths (/app/mlruns/...) instead of the host's absolute path
# (e.g. E:/MLOPS-01/mlruns/...), which would not resolve at runtime.
RUN python -m src.data.validate_data \
    && python -m src.features.build_features \
    && python -m src.models.train \
    && python -m src.models.evaluate

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
