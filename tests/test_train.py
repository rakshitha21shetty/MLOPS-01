import pandas as pd

from src.models.train import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_candidates,
    build_preprocessor,
    evaluate,
)


def _sample_features_df(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Item_Weight": [9.3, 5.9, 17.5, 8.9, 13.6, 6.4],
            "Item_Visibility": [0.016, 0.02, 0.05, 0.1, 0.03, 0.07],
            "Item_MRP": [249.8, 48.3, 141.6, 182.1, 53.9, 230.0],
            "Outlet_Age": [14, 4, 26, 9, 26, 11],
            "Item_Fat_Content": ["Low Fat", "Regular", "Low Fat", "Regular", "Non-Edible", "Low Fat"],
            "Item_Type": [
                "Dairy", "Soft Drinks", "Meat", "Household", "Household", "Snack Foods",
            ],
            "Item_Category": ["Food", "Drinks", "Food", "Non-Consumable", "Non-Consumable", "Food"],
            "Outlet_Size": ["Medium", "Small", "High", "Medium", "Small", "Medium"],
            "Outlet_Location_Type": ["Tier 1", "Tier 3", "Tier 2", "Tier 1", "Tier 3", "Tier 1"],
            "Outlet_Type": [
                "Supermarket Type1", "Grocery Store", "Supermarket Type3",
                "Supermarket Type1", "Grocery Store", "Supermarket Type2",
            ],
        }
    )[:n]


def test_preprocessor_transforms_expected_columns():
    X = _sample_features_df()
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(X)


def test_build_candidates_returns_three_models():
    params = {
        "model": {
            "random_state": 42,
            "random_forest": {"n_estimators": 10, "max_depth": 3},
            "xgboost": {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1},
        }
    }
    candidates = build_candidates(params)
    assert set(candidates.keys()) == {"linear_regression", "random_forest", "xgboost"}


def test_pipeline_fits_and_predicts():
    from sklearn.pipeline import Pipeline

    X = _sample_features_df()
    y = pd.Series([3735.1, 443.4, 2097.3, 732.4, 994.8, 556.6])

    pipeline = Pipeline(
        steps=[("preprocess", build_preprocessor()), ("model", build_candidates(
            {
                "model": {
                    "random_state": 42,
                    "random_forest": {"n_estimators": 10, "max_depth": 3},
                    "xgboost": {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1},
                }
            }
        )["linear_regression"])]
    )
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    assert len(predictions) == len(X)


def test_evaluate_returns_expected_metric_keys():
    y_true = [100, 200, 300]
    y_pred = [110, 190, 290]
    metrics = evaluate(y_true, y_pred)
    assert set(metrics.keys()) == {"rmse", "mae", "r2"}
    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0


def test_feature_lists_do_not_overlap():
    assert set(NUMERIC_FEATURES).isdisjoint(set(CATEGORICAL_FEATURES))
