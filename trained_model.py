import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

df = pd.read_csv("vehicle_data.csv")

# 👉 TẠO LABEL
df["label"] = (df["remaining_km"] <= 500).astype(int)

# 👉 FEATURE & LABEL
X = df.drop(["label", "remaining_km"], axis=1)
y = df["label"]

# 👉 TRAIN
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)
model.fit(X, y)

# 👉 Tạo thư mục nếu chưa tồn tại
os.makedirs("backend/model", exist_ok=True)

# 👉 SAVE
joblib.dump(model, "backend/model/rf_maintenance.pkl")

print("Train RandomForest thanh cong")
