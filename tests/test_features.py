import pandas as pd

from src.features.build_features import build_features


def _sample_raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Item_Identifier": "FDA15",
                "Item_Weight": 9.3,
                "Item_Fat_Content": "low fat",
                "Item_Visibility": 0.0,
                "Item_Type": "Dairy",
                "Item_MRP": 249.8,
                "Outlet_Identifier": "OUT049",
                "Outlet_Establishment_Year": 1999,
                "Outlet_Size": "Medium",
                "Outlet_Location_Type": "Tier 1",
                "Outlet_Type": "Supermarket Type1",
                "Item_Outlet_Sales": 3735.14,
            },
            {
                "Item_Identifier": "NCD19",
                "Item_Weight": None,
                "Item_Fat_Content": "Low Fat",
                "Item_Visibility": 0.04,
                "Item_Type": "Household",
                "Item_MRP": 53.86,
                "Outlet_Identifier": "OUT018",
                "Outlet_Establishment_Year": 2009,
                "Outlet_Size": None,
                "Outlet_Location_Type": "Tier 3",
                "Outlet_Type": "Supermarket Type2",
                "Item_Outlet_Sales": 443.42,
            },
        ]
    )


def test_build_features_has_no_nulls():
    df = _sample_raw_df()
    processed = build_features(df)
    assert processed.isna().sum().sum() == 0


def test_build_features_normalizes_fat_content():
    df = _sample_raw_df()
    processed = build_features(df)
    assert set(processed["Item_Fat_Content"]).issubset({"Low Fat", "Regular", "Non-Edible"})


def test_non_consumable_gets_non_edible_fat_content():
    df = _sample_raw_df()
    processed = build_features(df)
    household_row = processed[processed["Item_Category"] == "Non-Consumable"]
    assert (household_row["Item_Fat_Content"] == "Non-Edible").all()


def test_build_features_drops_identifier_columns():
    df = _sample_raw_df()
    processed = build_features(df)
    assert "Item_Identifier" not in processed.columns
    assert "Outlet_Identifier" not in processed.columns
    assert "Outlet_Establishment_Year" not in processed.columns


def test_outlet_age_is_positive():
    df = _sample_raw_df()
    processed = build_features(df)
    assert (processed["Outlet_Age"] > 0).all()
