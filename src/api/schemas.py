from typing import Literal

from pydantic import BaseModel, Field

ItemFatContent = Literal["Low Fat", "Regular", "Non-Edible"]

ItemType = Literal[
    "Baking Goods",
    "Breads",
    "Breakfast",
    "Canned",
    "Dairy",
    "Frozen Foods",
    "Fruits and Vegetables",
    "Hard Drinks",
    "Health and Hygiene",
    "Household",
    "Meat",
    "Others",
    "Seafood",
    "Snack Foods",
    "Soft Drinks",
    "Starchy Foods",
]

ItemCategory = Literal["Food", "Drinks", "Non-Consumable"]
OutletSize = Literal["Small", "Medium", "High"]
OutletLocationType = Literal["Tier 1", "Tier 2", "Tier 3"]
OutletType = Literal[
    "Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"
]


class PredictionRequest(BaseModel):
    Item_Weight: float = Field(..., gt=0, le=40, description="Item weight in kg")
    Item_Visibility: float = Field(
        ..., ge=0, le=1, description="Fraction of total display area allotted to this item"
    )
    Item_MRP: float = Field(..., gt=0, description="Maximum retail price")
    Outlet_Age: int = Field(..., ge=0, le=100, description="Years since the outlet opened")
    Item_Fat_Content: ItemFatContent
    Item_Type: ItemType
    Item_Category: ItemCategory
    Outlet_Size: OutletSize
    Outlet_Location_Type: OutletLocationType
    Outlet_Type: OutletType

    model_config = {
        "json_schema_extra": {
            "example": {
                "Item_Weight": 9.3,
                "Item_Visibility": 0.016,
                "Item_MRP": 249.8,
                "Outlet_Age": 14,
                "Item_Fat_Content": "Low Fat",
                "Item_Type": "Dairy",
                "Item_Category": "Food",
                "Outlet_Size": "Medium",
                "Outlet_Location_Type": "Tier 1",
                "Outlet_Type": "Supermarket Type1",
            }
        }
    }


class PredictionResponse(BaseModel):
    predicted_sales: float
    model_name: str
    model_version: str
    latency_ms: float
