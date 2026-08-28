"""Train and compare candidate models for BigMart sales prediction.

Logs every run (params, CV + holdout metrics, the fitted pipeline) to
MLflow. Each model is a single sklearn Pipeline (preprocessing +
regressor) so the exact same object can be saved, loaded, and served
without re-implementing encoding logic at inference time.
"""

import numpy as np
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.config import ROOT_DIR, get_mlflow_tracking_uri, load_params

NUMERIC_FEATURES = ["Item_Weight", "Item_Visibility", "Item_MRP", "Outlet_Age"]
CATEGORICAL_FEATURES = [
    "Item_Fat_Content",
    "Item_Type",
    "Item_Category",
    "Outlet_Size",
    "Outlet_Location_Type",
    "Outlet_Type",
]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_candidates(params: dict) -> dict:
    model_params = params["model"]
    rf_params = model_params["random_forest"]
    xgb_params = model_params["xgboost"]
    random_state = model_params["random_state"]

    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=rf_params["n_estimators"],
            max_depth=rf_params["max_depth"],
            random_state=random_state,
        ),
        "xgboost": XGBRegressor(
            n_estimators=xgb_params["n_estimators"],
            max_depth=xgb_params["max_depth"],
            learning_rate=xgb_params["learning_rate"],
            random_state=random_state,
        ),
    }


def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_all(params: dict) -> list[dict]:
    data_params = params["data"]
    processed_path = ROOT_DIR / data_params["processed_path"]
    df = pd.read_csv(processed_path)

    target = data_params["target"]
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=data_params["test_size"], random_state=data_params["random_state"]
    )

    mlflow_params = params["mlflow"]
    mlflow.set_tracking_uri(get_mlflow_tracking_uri(params))
    mlflow.set_experiment(mlflow_params["experiment_name"])

    cv = KFold(n_splits=params["model"]["cv_folds"], shuffle=True, random_state=42)
    candidates = build_candidates(params)
    results = []

    for name, estimator in candidates.items():
        pipeline = Pipeline(
            steps=[("preprocess", build_preprocessor()), ("model", estimator)]
        )

        with mlflow.start_run(run_name=name) as run:
            cv_scores = cross_val_score(
                pipeline, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error"
            )
            cv_rmse = float(-cv_scores.mean())

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            holdout_metrics = evaluate(y_test, y_pred)

            mlflow.log_param("model_name", name)
            mlflow.log_metric("cv_rmse", cv_rmse)
            for metric_name, value in holdout_metrics.items():
                mlflow.log_metric(f"holdout_{metric_name}", value)

            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBRegressor"],
            )

            results.append(
                {
                    "run_id": run.info.run_id,
                    "model_name": name,
                    "cv_rmse": cv_rmse,
                    **{f"holdout_{k}": v for k, v in holdout_metrics.items()},
                }
            )
            print(f"[{name}] cv_rmse={cv_rmse:.2f} holdout={holdout_metrics}")

    return results


def main() -> None:
    params = load_params()
    train_all(params)


if __name__ == "__main__":
    main()
