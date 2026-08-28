import time
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient

from src.api.schemas import PredictionRequest, PredictionResponse
from src.config import get_mlflow_tracking_uri, load_params
from src.models.evaluate import PRODUCTION_ALIAS

model_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    params = load_params()
    mlflow.set_tracking_uri(get_mlflow_tracking_uri(params))

    registered_name = params["mlflow"]["registered_model_name"]
    model_uri = f"models:/{registered_name}@{PRODUCTION_ALIAS}"

    client = MlflowClient()
    version_info = client.get_model_version_by_alias(registered_name, PRODUCTION_ALIAS)

    model_state["model"] = mlflow.pyfunc.load_model(model_uri)
    model_state["model_name"] = registered_name
    model_state["model_version"] = version_info.version

    yield
    model_state.clear()


app = FastAPI(title="BigMart Sales Forecasting API", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok" if "model" in model_state else "model not loaded",
        "model_name": model_state.get("model_name"),
        "model_version": model_state.get("model_version"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if "model" not in model_state:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    start = time.perf_counter()
    input_df = pd.DataFrame([request.model_dump()])
    prediction = model_state["model"].predict(input_df)[0]
    latency_ms = (time.perf_counter() - start) * 1000

    return PredictionResponse(
        predicted_sales=round(float(prediction), 2),
        model_name=model_state["model_name"],
        model_version=str(model_state["model_version"]),
        latency_ms=round(latency_ms, 2),
    )
