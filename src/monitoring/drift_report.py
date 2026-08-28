"""Generate a data-drift HTML report comparing a reference slice of the
training data against a held-out "current" slice, simulating incoming
production data. Uses Evidently's DataDriftPreset.
"""

import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset
from sklearn.model_selection import train_test_split

from src.config import ROOT_DIR, load_params
from src.models.train import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_data_definition() -> DataDefinition:
    return DataDefinition(
        numerical_columns=NUMERIC_FEATURES,
        categorical_columns=CATEGORICAL_FEATURES,
    )


def generate_drift_report(df: pd.DataFrame, params: dict):
    monitoring_params = params["monitoring"]
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    reference_df, current_df = train_test_split(
        df[feature_cols],
        train_size=monitoring_params["reference_sample_size"],
        random_state=42,
    )

    data_definition = build_data_definition()
    reference_dataset = Dataset.from_pandas(reference_df, data_definition=data_definition)
    current_dataset = Dataset.from_pandas(current_df, data_definition=data_definition)

    report = Report(metrics=[DataDriftPreset()])
    return report.run(current_data=current_dataset, reference_data=reference_dataset)


def main() -> None:
    params = load_params()
    processed_path = ROOT_DIR / params["data"]["processed_path"]
    reports_dir = ROOT_DIR / params["monitoring"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(processed_path)
    snapshot = generate_drift_report(df, params)

    output_path = reports_dir / "drift_report.html"
    snapshot.save_html(str(output_path))
    print(f"Drift report written to {output_path}")


if __name__ == "__main__":
    main()
