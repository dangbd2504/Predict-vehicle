import joblib
import pandas as pd

# Tải mô hình
model_path = "backend/model/rf_maintenance_new.pkl"
model = joblib.load(model_path)

print("Model loaded successfully!")
print(f"Model type: {type(model)}")

# Tạo dữ liệu thử nghiệm
test_data = pd.DataFrame([{
    "total_km": 15000,
    "avg_km_per_trip": 20,
    "trips_per_day": 4,
    "vehicle_age_months": 12
}])

print("Test data:")
print(test_data)

# Dự đoán
try:
    prediction = model.predict(test_data)[0]
    print(f"Prediction: {prediction}")
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(test_data)[0]
        print(f"Probabilities: {probabilities}")
    else:
        print("Model does not support predict_proba")
        
except Exception as e:
    print(f"Error during prediction: {e}")
    import traceback
    traceback.print_exc()