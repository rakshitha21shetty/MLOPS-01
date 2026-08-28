"""Streamlit dashboard: latest model metrics, drift report, and a live
prediction form against the registered production model.

Run with: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import mlflow
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from mlflow import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT_DIR, get_mlflow_tracking_uri, load_params  # noqa: E402
from src.models.evaluate import PRODUCTION_ALIAS  # noqa: E402
from src.models.train import CATEGORICAL_FEATURES, NUMERIC_FEATURES  # noqa: E402
from src.monitoring.drift_report import generate_drift_report  # noqa: E402

st.set_page_config(page_title="BigMart Sales Forecasting", layout="wide")

params = load_params()
mlflow.set_tracking_uri(get_mlflow_tracking_uri(params))
client = MlflowClient()

st.title("BigMart Sales Forecasting — Monitoring Dashboard")

# --- Model metrics ---------------------------------------------------
st.header("Registered Model")

registered_name = params["mlflow"]["registered_model_name"]
try:
    version_info = client.get_model_version_by_alias(registered_name, PRODUCTION_ALIAS)
    run = client.get_run(version_info.run_id)
    metrics = run.data.metrics
    params_logged = run.data.params

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", params_logged.get("model_name", "n/a"))
    col2.metric("Version", version_info.version)
    col3.metric("Holdout RMSE", f"{metrics.get('holdout_rmse', 0):.2f}")
    col4.metric("Holdout R2", f"{metrics.get('holdout_r2', 0):.3f}")
except Exception as e:
    st.warning(f"No production model found yet. Run the training pipeline first. ({e})")

# --- Data drift report ------------------------------------------------
st.header("Data Drift Report")
processed_path = ROOT_DIR / params["data"]["processed_path"]

if processed_path.exists():
    if st.button("Generate / refresh drift report"):
        df = pd.read_csv(processed_path)
        snapshot = generate_drift_report(df, params)
        reports_dir = ROOT_DIR / params["monitoring"]["reports_dir"]
        reports_dir.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(reports_dir / "drift_report.html"))
        st.success("Drift report refreshed.")

    report_path = ROOT_DIR / params["monitoring"]["reports_dir"] / "drift_report.html"
    if report_path.exists():
        html = report_path.read_text(encoding="utf-8")
        components.html(html, height=800, scrolling=True)
    else:
        st.info("No drift report yet — click the button above to generate one.")
else:
    st.info("No processed data found yet. Run the feature pipeline first.")

# --- Live prediction ---------------------------------------------------
st.header("Try a Prediction")

with st.form("predict_form"):
    c1, c2 = st.columns(2)
    with c1:
        item_weight = st.number_input("Item Weight", min_value=0.1, max_value=40.0, value=9.3)
        item_visibility = st.slider("Item Visibility", 0.0, 1.0, 0.016)
        item_mrp = st.number_input("Item MRP", min_value=1.0, value=249.8)
        outlet_age = st.number_input("Outlet Age (years)", min_value=0, max_value=100, value=14)
    with c2:
        item_fat_content = st.selectbox("Item Fat Content", ["Low Fat", "Regular", "Non-Edible"])
        item_type = st.selectbox(
            "Item Type",
            [
                "Baking Goods", "Breads", "Breakfast", "Canned", "Dairy", "Frozen Foods",
                "Fruits and Vegetables", "Hard Drinks", "Health and Hygiene", "Household",
                "Meat", "Others", "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods",
            ],
        )
        item_category = st.selectbox("Item Category", ["Food", "Drinks", "Non-Consumable"])
        outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
        outlet_location_type = st.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
        outlet_type = st.selectbox(
            "Outlet Type",
            ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"],
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    row = pd.DataFrame(
        [
            {
                "Item_Weight": item_weight,
                "Item_Visibility": item_visibility,
                "Item_MRP": item_mrp,
                "Outlet_Age": outlet_age,
                "Item_Fat_Content": item_fat_content,
                "Item_Type": item_type,
                "Item_Category": item_category,
                "Outlet_Size": outlet_size,
                "Outlet_Location_Type": outlet_location_type,
                "Outlet_Type": outlet_type,
            }
        ]
    )
    assert set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) == set(row.columns)

    model_uri = f"models:/{registered_name}@{PRODUCTION_ALIAS}"
    model = mlflow.pyfunc.load_model(model_uri)
    prediction = model.predict(row)[0]
    st.success(f"Predicted sales: **{prediction:.2f}**")
