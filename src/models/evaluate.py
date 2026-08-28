"""Pick the best run from the latest training experiment, register it in
the MLflow Model Registry, and promote it with a 'production' alias.

MLflow 2.9+ deprecated registry stages (Staging/Production) in favor of
aliases, so we use `set_registered_model_alias` rather than
`transition_model_version_stage`.
"""

import mlflow
from mlflow import MlflowClient

from src.config import get_mlflow_tracking_uri, load_params

PRODUCTION_ALIAS = "production"


def get_best_run(client: MlflowClient, experiment_name: str):
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.holdout_rmse ASC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")
    return runs[0]


def register_best_model(params: dict) -> str:
    mlflow.set_tracking_uri(get_mlflow_tracking_uri(params))
    client = MlflowClient()

    mlflow_params = params["mlflow"]
    experiment_name = mlflow_params["experiment_name"]
    registered_name = mlflow_params["registered_model_name"]

    best_run = get_best_run(client, experiment_name)
    run_id = best_run.info.run_id
    model_name = best_run.data.params.get("model_name", "unknown")
    holdout_rmse = best_run.data.metrics.get("holdout_rmse")

    model_uri = f"runs:/{run_id}/model"
    model_version = mlflow.register_model(model_uri=model_uri, name=registered_name)

    client.set_registered_model_alias(
        name=registered_name, alias=PRODUCTION_ALIAS, version=model_version.version
    )

    print(
        f"Registered '{registered_name}' v{model_version.version} "
        f"(source model: {model_name}, run: {run_id}, holdout_rmse: {holdout_rmse:.2f}) "
        f"and aliased it as '@{PRODUCTION_ALIAS}'."
    )
    return f"models:/{registered_name}@{PRODUCTION_ALIAS}"


def main() -> None:
    params = load_params()
    register_best_model(params)


if __name__ == "__main__":
    main()
