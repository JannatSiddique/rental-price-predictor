import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from preprocessing import preprocess 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "zameen_rentals_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "rent_predictor.pkl")

df = pd.read_csv(DATA_PATH)
df, numeric_cols = preprocess(df)

X = df.drop(columns=["rent"])
y = df["rent"]

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X, y)

preds = model.predict(X)
mae = mean_absolute_error(y, preds)
print(f"MAE: {mae:,.0f}")

joblib.dump(
    {
        "model": model,
        "features": X.columns.tolist(),
        "scaler": scaler,
        "numeric_cols": numeric_cols
    },
    MODEL_PATH
)

print("Model, features, and scaler saved successfully.")
