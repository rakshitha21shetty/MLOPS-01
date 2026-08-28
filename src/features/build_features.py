"""Cleaning and feature engineering for the BigMart sales dataset.

Produces a cleaned CSV with consistent categories, imputed nulls, and a
couple of derived features. Categorical encoding is deliberately left to
the sklearn Pipeline in src/models/train.py so the same fitted
preprocessing travels with the model artifact into serving.
"""

import numpy as np
import pandas as pd

from src.config import ROOT_DIR, load_params

# Dataset was collected in 2013 (per the Kaggle dataset description);
# outlet age is computed relative to that reference year.
REFERENCE_YEAR = 2013

FAT_CONTENT_MAP = {
    "low fat": "Low Fat",
    "LF": "Low Fat",
    "Low Fat": "Low Fat",
    "reg": "Regular",
    "Regular": "Regular",
}

ITEM_CATEGORY_MAP = {
    "FD": "Food",
    "DR": "Drinks",
    "NC": "Non-Consumable",
}


def clean_fat_content(df: pd.DataFrame) -> pd.DataFrame:
    df["Item_Fat_Content"] = df["Item_Fat_Content"].map(FAT_CONTENT_MAP)
    return df


def add_item_category(df: pd.DataFrame) -> pd.DataFrame:
    df["Item_Category"] = df["Item_Identifier"].str[:2].map(ITEM_CATEGORY_MAP)
    # Non-consumables (household, hygiene, etc.) don't have a meaningful
    # fat content — relabel rather than leaving a misleading value.
    df.loc[df["Item_Category"] == "Non-Consumable", "Item_Fat_Content"] = "Non-Edible"
    return df


def impute_item_weight(df: pd.DataFrame) -> pd.DataFrame:
    # Same Item_Identifier should have the same weight across outlets.
    df["Item_Weight"] = df.groupby("Item_Identifier")["Item_Weight"].transform(
        lambda s: s.fillna(s.mean())
    )
    df["Item_Weight"] = df["Item_Weight"].fillna(df["Item_Weight"].mean())
    return df


def impute_item_visibility(df: pd.DataFrame) -> pd.DataFrame:
    # A visibility of exactly 0 is physically implausible for a stocked
    # item — treat it as missing and impute by item type average.
    df["Item_Visibility"] = df["Item_Visibility"].replace(0, np.nan)
    df["Item_Visibility"] = df.groupby("Item_Type")["Item_Visibility"].transform(
        lambda s: s.fillna(s.mean())
    )
    # Falls back to the overall mean if an entire Item_Type group had no
    # non-zero visibility to average (e.g. a sparse batch of new data).
    df["Item_Visibility"] = df["Item_Visibility"].fillna(df["Item_Visibility"].mean())
    return df


def impute_outlet_size(df: pd.DataFrame) -> pd.DataFrame:
    mode_by_type = df.groupby("Outlet_Type")["Outlet_Size"].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else pd.NA
    )
    overall_mode = df["Outlet_Size"].mode().iloc[0]

    def fill(row):
        if pd.notna(row["Outlet_Size"]):
            return row["Outlet_Size"]
        candidate = mode_by_type.get(row["Outlet_Type"])
        return candidate if pd.notna(candidate) else overall_mode

    df["Outlet_Size"] = df.apply(fill, axis=1)
    return df


def add_outlet_age(df: pd.DataFrame) -> pd.DataFrame:
    df["Outlet_Age"] = REFERENCE_YEAR - df["Outlet_Establishment_Year"]
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = clean_fat_content(df)
    df = add_item_category(df)
    df = impute_item_weight(df)
    df = impute_item_visibility(df)
    df = impute_outlet_size(df)
    df = add_outlet_age(df)
    df = df.drop(columns=["Item_Identifier", "Outlet_Identifier", "Outlet_Establishment_Year"])
    return df


def main() -> None:
    params = load_params()
    raw_path = ROOT_DIR / params["data"]["raw_train_path"]
    processed_path = ROOT_DIR / params["data"]["processed_path"]
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)
    processed = build_features(df)
    processed.to_csv(processed_path, index=False)
    print(f"Wrote {len(processed)} rows, {len(processed.columns)} columns to {processed_path}")


if __name__ == "__main__":
    main()
