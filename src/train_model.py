import os
import re
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "zameen_rentals_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "rent_predictor.pkl")

df = pd.read_csv(DATA_PATH)

# --- CONVERT PRICE ---
def convert_price(x):
    if pd.isna(x):
        return None
    x = str(x).strip().lower().replace(",", "")
    match = re.match(r"([\d.]+)\s*(lakh|thousand|crore)?", x)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    if unit == "lakh":
        return value * 100_000
    elif unit == "thousand":
        return value * 1_000
    elif unit == "crore":
        return value * 10_000_000
    return value

df["Price"] = df["Price"].apply(convert_price)

# --- EXTRACT MAIN LOCATION ---
def extract_location(loc):
    if pd.isna(loc):
        return "Unknown"
    parts = str(loc).split(",")
    return parts[-1].strip().title()

df["Location"] = df["Location"].apply(extract_location)

# --- KEEP ONLY COMMON LOCATIONS (50+ listings) ---
location_counts = df["Location"].value_counts()
df["Location"] = df["Location"].apply(
    lambda x: x if location_counts.get(x, 0) >= 50 else "Other"
)
print(f"✅ Locations after filtering: {df['Location'].nunique()}")

# --- CONVERT AREA ---
def convert_area(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    if "Kanal" in x:
        return float(x.replace("Kanal", "").strip()) * 20
    elif "Marla" in x:
        return float(x.replace("Marla", "").strip())
    else:
        try:
            return float(x)
        except:
            return None

df["Marla"] = df["Marla"].apply(convert_area)
df["Bedrooms"] = pd.to_numeric(df["Bedrooms"], errors="coerce")
df["Washrooms"] = pd.to_numeric(df["Washrooms"], errors="coerce")

# --- CLEAN ---
df = df.dropna(subset=["Bedrooms", "Washrooms", "Marla", "Price", "Location"])
Q1 = df["Price"].quantile(0.05)
Q3 = df["Price"].quantile(0.95)
df = df[(df["Price"] >= Q1) & (df["Price"] <= Q3)]

print(f"✅ Rows after cleaning: {len(df)}")
print(f"✅ Price range: PKR {df['Price'].min():,.0f} → PKR {df['Price'].max():,.0f}")
print(df["Location"].value_counts().head(10))

# --- ONE-HOT ENCODE ---
df = pd.get_dummies(df, columns=["Location"], prefix="Location")

numeric_cols = ["Bedrooms", "Washrooms", "Marla"]
location_cols = [col for col in df.columns if col.startswith("Location_")]
features = numeric_cols + location_cols

X = df[features].copy()
y = df["Price"]

# --- SCALE ---
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# --- TRAIN (faster settings) ---
print("⏳ Training model...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    n_jobs=-1,
    random_state=42
)
model.fit(X, y)

# --- VERIFY ---
preds = model.predict(X)
mae = mean_absolute_error(y, preds)
print(f"📊 MAE: PKR {mae:,.0f}")

# Quick sanity check
test_row = {col: 0 for col in features}
test_row["Bedrooms"] = 3
test_row["Washrooms"] = 2
test_row["Marla"] = 5
if "Location_Dha Defence" in features:
    test_row["Location_Dha Defence"] = 1

test_df = pd.DataFrame([test_row])
test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])
print(f"🧪 Test prediction (3bed/2bath/5marla/DHA): PKR {model.predict(test_df)[0]:,.0f}")

# --- SAVE ---
joblib.dump({
    "model": model,
    "features": features,
    "scaler": scaler,
    "numeric_cols": numeric_cols
}, MODEL_PATH)

print("✅ Model saved successfully.")