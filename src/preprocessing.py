import pandas as pd
import numpy as np

def clean_price(price):
    if isinstance(price, str):
        price = price.strip()
        if "Thousand" in price:
            return float(price.replace("Thousand", "").strip()) * 1000
        if "Lakh" in price:
            return float(price.replace("Lakh", "").strip()) * 100000
    return np.nan

def preprocess(df: pd.DataFrame):
    """
    Preprocess the Zameen rental dataset:
    - Convert price strings to numeric 'rent'
    - Handle numeric columns and missing values
    - Encode categorical variables
    - Convert 'Details' into numeric feature
    Returns:
        df: preprocessed DataFrame
        numeric_cols: list of numeric columns for scaling
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    if "Price" in df.columns:
        df.rename(columns={"Price": "rent"}, inplace=True)

    if "rent" in df.columns:
        df["rent"] = df["rent"].apply(clean_price)
        df = df.dropna(subset=["rent"])

    numeric_cols = ["Bedrooms", "Washrooms", "Marla"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    if "Details" in df.columns:
        df["Details_length"] = df["Details"].astype(str).apply(len)
    else:
        df["Details_length"] = 0

    numeric_cols.append("Details_length")

    df["Location"] = df["Location"].fillna("Other")
    df["Currency"] = df["Currency"].fillna("PKR")

    if "rent" in df.columns:
        top_locations = df["Location"].value_counts().nlargest(20).index
        df["Location"] = df["Location"].where(df["Location"].isin(top_locations), "Other")

    df = pd.get_dummies(df, columns=["Location", "Currency"], drop_first=False)

    return df, numeric_cols
