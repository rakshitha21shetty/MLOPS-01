from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent


def load_params(params_path: Path | str = ROOT_DIR / "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def get_mlflow_tracking_uri(params: dict) -> str:
    # Local SQLite backend: the plain filesystem store is in maintenance
    # mode in current MLflow and doesn't support the Model Registry.
    db_path = ROOT_DIR / params["mlflow"]["tracking_uri"] / "mlflow.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path.as_posix()}"
