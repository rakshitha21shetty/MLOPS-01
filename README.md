# BigMart Sales Forecasting — MLOps Pipeline

A sample end-to-end MLOps project: train, track, version, deploy, and monitor a
regression model that predicts per-item sales for a retail chain, using only
free/open-source tools and a free dataset.

**Dataset**: [BigMart Sales Data](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data)
(Kaggle, free) — ~8,500 rows predicting `Item_Outlet_Sales` from item and outlet
attributes.

## Stack

| Concern | Tool |
|---|---|
| Data versioning / pipeline orchestration | DVC (local remote — no cloud storage) |
| Data validation | pandera |
| Training | scikit-learn, XGBoost |
| Experiment tracking / model registry | MLflow (local SQLite backend) |
| Serving | FastAPI + Uvicorn |
| Containerization | Docker |
| CI | GitHub Actions |
| Drift monitoring | Evidently |
| Dashboard | Streamlit |

## One-time setup

1. **Python env**
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate   # Windows Git Bash; use .venv\Scripts\activate on cmd
   pip install -r requirements.txt
   ```

2. **Kaggle API token** (free — needed to download the dataset)
   - Sign up at [kaggle.com](https://www.kaggle.com) (free).
   - Go to **Settings → API → Create New Token**.
   - Newer Kaggle accounts get a token like `KGAT_...` — save it as plain text
     in `~/.kaggle/access_token`. Older accounts get a `kaggle.json` file
     (`{"username": ..., "key": ...}`) — save that at `~/.kaggle/kaggle.json`
     instead. Either is picked up automatically by `kagglehub`.
   - **Never commit either file** — both paths are already gitignored.

3. **DVC remote** (local folder — no paid cloud storage)
   ```bash
   dvc remote add -d localstorage ../mlops-01-dvc-storage
   ```

## Running the pipeline

Run the whole thing with DVC:

```bash
dvc repro
```

This chains 5 stages: `download → validate → build_features → train → evaluate`.
Or run each stage manually:

```bash
python -m src.data.download_data       # pulls the dataset from Kaggle
python -m src.data.validate_data       # pandera schema + null-rate checks
python -m src.features.build_features  # cleaning + feature engineering
python -m src.models.train             # trains 3 candidate models, logs to MLflow
python -m src.models.evaluate          # registers the best run, aliases it @production
```

Inspect experiments:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow_store/mlflow.db
```

## Serving

Locally:
```bash
uvicorn src.api.main:app --reload
```
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "Item_Weight": 9.3, "Item_Visibility": 0.016, "Item_MRP": 249.8, "Outlet_Age": 14,
  "Item_Fat_Content": "Low Fat", "Item_Type": "Dairy", "Item_Category": "Food",
  "Outlet_Size": "Medium", "Outlet_Location_Type": "Tier 1", "Outlet_Type": "Supermarket Type1"
}'
```

With Docker (trains the model *inside* the image build, so the MLflow registry
stores container-native paths rather than host paths):
```bash
docker build -t bigmart-sales-api .
docker run -p 8000:8000 bigmart-sales-api
```

Or via Compose (also starts an `mlflow ui` service on port 5000):
```bash
docker compose up --build
```

## Monitoring

```bash
python -m src.monitoring.drift_report   # writes reports/drift_report.html
streamlit run dashboard/app.py          # metrics + drift report + live prediction form
```

## Tests

```bash
pytest tests/ -v
```
API tests stub the loaded model directly (no trained model or Kaggle access
needed), so they run standalone in CI. Feature/training tests use small
synthetic dataframes for the same reason.

## Project layout

```
src/
  data/            download + pandera validation
  features/        cleaning & feature engineering
  models/          train (MLflow logging) + evaluate (registry promotion)
  monitoring/       Evidently drift report
  api/             FastAPI serving app
dashboard/         Streamlit monitoring UI
tests/             pytest suite
dvc.yaml           pipeline definition
params.yaml        central config
```
