import joblib
import pandas as pd


model_path = "backend/model/rf_maintenance_new.pkl"
model = joblib.load(model_path)

print("Mô hình đã được tải thành công!")
print(f"Loại mô hình: {type(model)}")

# Tạo dữ liệu thử nghiệm
test_data = pd.DataFrame([{
    "total_km": 15000,
    "avg_km_per_trip": 20,
    "trips_per_day": 4,
    "vehicle_age_months": 12
}])

print("Dữ liệu thử nghiệm:")
print(test_data)

# Dự đoán
try:
    prediction = model.predict(test_data)[0]
    print(f"Dự đoán: {prediction}")
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(test_data)[0]
        print(f"Xác suất: {probabilities}")
    else:
        print("Mô hình không hỗ trợ predict_proba")
        
except Exception as e:
    print(f"Lỗi khi dự đoán: {e}")
    import traceback
    traceback.print_exc()