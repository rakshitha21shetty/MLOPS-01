"""Data validation gate for the raw BigMart dataset.

Runs before feature engineering / training. Fails fast (raises) if the
incoming data doesn't match the expected schema or if null rates blow
past sane thresholds — a lightweight guardrail against silently
training on corrupted or drifted upstream data.
"""

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

from src.config import ROOT_DIR, load_params

# Null-rate ceilings: if more than this fraction of a column is missing,
# something upstream likely broke (vs. the dataset's normal missingness).
MAX_NULL_RATE = {
    "Item_Weight": 0.30,
    "Outlet_Size": 0.35,
}

RAW_SCHEMA = DataFrameSchema(
    {
        "Item_Identifier": Column(str, nullable=False),
        "Item_Weight": Column(float, Check.in_range(0, 40), nullable=True),
        "Item_Fat_Content": Column(
            str, Check.isin(["Low Fat", "Regular", "low fat", "LF", "reg"])
        ),
        "Item_Visibility": Column(float, Check.in_range(0, 1)),
        "Item_Type": Column(str),
        "Item_MRP": Column(float, Check.greater_than(0)),
        "Outlet_Identifier": Column(str),
        "Outlet_Establishment_Year": Column(int, Check.in_range(1985, 2025)),
        "Outlet_Size": Column(str, Check.isin(["Small", "Medium", "High"]), nullable=True),
        "Outlet_Location_Type": Column(str, Check.isin(["Tier 1", "Tier 2", "Tier 3"])),
        "Outlet_Type": Column(
            str,
            Check.isin(
                ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"]
            ),
        ),
        "Item_Outlet_Sales": Column(float, Check.greater_than_or_equal_to(0)),
    },
    strict=False,
    coerce=True,
)


def check_null_rates(df: pd.DataFrame) -> None:
    for col, max_rate in MAX_NULL_RATE.items():
        null_rate = df[col].isna().mean()
        if null_rate > max_rate:
            raise ValueError(
                f"Null rate for '{col}' is {null_rate:.1%}, exceeding the "
                f"{max_rate:.0%} threshold. Refusing to proceed with training."
            )


def validate_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    validated = RAW_SCHEMA.validate(df, lazy=True)
    check_null_rates(validated)
    return validated


def main() -> None:
    params = load_params()
    raw_path = ROOT_DIR / params["data"]["raw_train_path"]
    df = pd.read_csv(raw_path)
    validate_raw_data(df)
    print(f"Validation passed: {len(df)} rows, {len(df.columns)} columns from {raw_path}")


if __name__ == "__main__":
    main()
