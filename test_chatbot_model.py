import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'prediction_system'))

from advanced_chatbot import AdvancedVehicleMaintenanceChatbot

# Tạo chatbot
chatbot = AdvancedVehicleMaintenanceChatbot()

# Kiểm tra xem mô hình có được tải không
if chatbot.model is not None:
    print("Chatbot model loaded successfully!")
    print(f"Model type: {type(chatbot.model)}")
    
    # Thử dự đoán với dữ liệu mẫu
    try:
        prediction, probability = chatbot.get_maintenance_status(15000, 20, 4, 12)
        print(f"Prediction: {prediction}")
        print(f"Probability: {probability}")
        
        if prediction is not None:
            status_msg = chatbot.get_status_message(prediction)
            print(f"Status: {status_msg}")
        else:
            print("Prediction returned None")
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
else:
    print("Chatbot model failed to load!")