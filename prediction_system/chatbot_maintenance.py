import joblib
import pandas as pd
import json
import re
from datetime import datetime

class VehicleMaintenanceChatbot:
    def __init__(self, model_path="maintenance_model.pkl"):
        """
        Khởi tạo chatbot với mô hình đã huấn luyện
        """
        self.model_path = model_path
        self.model = self.load_model()
        self.conversation_state = {}
        self.expected_inputs = [
            'total_km', 'avg_km_per_trip', 'trips_per_day', 'vehicle_age_months'
        ]
        
    def load_model(self):
        """
        Tải mô hình từ file
        """
        try:
            model = joblib.load(self.model_path)
            print(f"✅ Đã tải mô hình thành công từ: {self.model_path}")
            return model
        except Exception as e:
            print(f"❌ Lỗi khi tải mô hình: {e}")
            return None
    
    def preprocess_input(self, user_input):
        """
        Tiền xử lý đầu vào của người dùng
        """
        # Chuyển sang chữ thường và loại bỏ khoảng trắng dư
        user_input = user_input.lower().strip()
        
        # Trích xuất số từ chuỗi đầu vào
        numbers = re.findall(r'\d+\.?\d*', user_input)
        return numbers
    
    def get_remaining_km(self, total_km, avg_km_per_trip, trips_per_day, vehicle_age_months):
        """
        Dự đoán số km còn lại đến kỳ bảo dưỡng
        """
        if self.model is None:
            return None
            
        # Tạo DataFrame từ dữ liệu đầu vào
        input_data = pd.DataFrame([{
            "total_km": float(total_km),
            "avg_km_per_trip": float(avg_km_per_trip),
            "trips_per_day": int(trips_per_day),
            "vehicle_age_months": int(vehicle_age_months)
        }])
        
        # Dự đoán
        remaining_km = self.model.predict(input_data)[0]
        return remaining_km
    
    def get_status_message(self, remaining_km):
        """
        Trả về thông báo trạng thái dựa trên số km còn lại
        """
        if remaining_km <= 500:
            return "⚠️ CẢNH BÁO: Xe sắp cần bảo dưỡng!"
        else:
            return "✅ Xe vẫn đang hoạt động bình thường"
    
    def generate_response(self, user_input):
        """
        Sinh phản hồi dựa trên đầu vào của người dùng
        """
        user_input_lower = user_input.lower()
        
        # Phản hồi chào mừng
        if any(greeting in user_input_lower for greeting in ['xin chào', 'chào', 'hello', 'hi']):
            return ("🤖 Xin chào! Tôi là Chatbot dự đoán bảo dưỡng xe máy.\n\n"
                   "Tôi có thể giúp bạn:\n"
                   "- Dự đoán thời điểm cần bảo dưỡng xe\n"
                   "- Kiểm tra tình trạng xe\n\n"
                   "Vui lòng cung cấp các thông tin sau:\n"
                   "1. Tổng km đã chạy\n"
                   "2. Km trung bình mỗi chuyến\n"
                   "3. Số chuyến/ngày\n"
                   "4. Tuổi xe (tháng)")
        
        # Phản hồi thông tin hỗ trợ
        elif any(info in user_input_lower for info in ['giúp', 'hướng dẫn', 'trợ giúp', 'help']):
            return ("ℹ️ Tôi có thể giúp bạn dự đoán thời điểm cần bảo dưỡng xe máy.\n\n"
                   "Vui lòng cung cấp các thông tin sau:\n"
                   "1. Tổng km đã chạy\n"
                   "2. Km trung bình mỗi chuyến\n"
                   "3. Số chuyến/ngày\n"
                   "4. Tuổi xe (tháng)\n\n"
                   "Ví dụ: Tôi đã chạy 15000 km, trung bình mỗi chuyến 20 km, đi 4 chuyến mỗi ngày, xe đã 12 tháng tuổi")
        
        # Kiểm tra xem người dùng có cung cấp đủ thông tin không
        numbers = self.preprocess_input(user_input)
        
        if len(numbers) >= 4:
            try:
                # Lấy 4 số đầu tiên theo thứ tự mong muốn
                total_km = float(numbers[0])
                avg_km_per_trip = float(numbers[1])
                trips_per_day = int(float(numbers[2]))
                vehicle_age_months = int(float(numbers[3]))
                
                # Dự đoán
                remaining_km = self.get_remaining_km(
                    total_km, avg_km_per_trip, trips_per_day, vehicle_age_months
                )
                
                if remaining_km is not None:
                    status_msg = self.get_status_message(remaining_km)
                    
                    response = (
                        f"🔍 **KẾT QUẢ DỰ ĐOÁN**\n\n"
                        f"📊 Thông tin xe:\n"
                        f"- Tổng km đã chạy: {total_km} km\n"
                        f"- Km trung bình mỗi chuyến: {avg_km_per_trip} km\n"
                        f"- Số chuyến/ngày: {trips_per_day}\n"
                        f"- Tuổi xe: {vehicle_age_months} tháng\n\n"
                        f"📈 Dự đoán:\n"
                        f"- Còn khoảng **{round(remaining_km)} km** đến kỳ bảo dưỡng\n\n"
                        f"🔔 Trạng thái: {status_msg}"
                    )
                    
                    return response
                else:
                    return "❌ Rất tiếc, không thể thực hiện dự đoán. Vui lòng kiểm tra lại mô hình."
                    
            except ValueError:
                return ("❌ Dữ liệu không hợp lệ. Vui lòng cung cấp:\n"
                       "1. Tổng km đã chạy\n"
                       "2. Km trung bình mỗi chuyến\n"
                       "3. Số chuyến/ngày\n"
                       "4. Tuổi xe (tháng)")
        
        # Nếu không có đủ thông tin, yêu cầu người dùng cung cấp
        else:
            return ("🤔 Vui lòng cung cấp đầy đủ thông tin:\n"
                   "• Tổng km đã chạy\n"
                   "• Km trung bình mỗi chuyến\n"
                   "• Số chuyến/ngày\n"
                   "• Tuổi xe (tháng)\n\n"
                   "Ví dụ: Tôi đã chạy 15000 km, trung bình mỗi chuyến 20 km, đi 4 chuyến mỗi ngày, xe đã 12 tháng tuổi")
    
    def chat(self, user_input):
        """
        Hàm chính để trò chuyện với chatbot
        """
        if not user_input.strip():
            return "🤖 Xin vui lòng nhập câu hỏi hoặc thông tin của bạn."
        
        response = self.generate_response(user_input)
        return response

# Hàm demo
def demo_chatbot():
    """
    Hàm demo để thử nghiệm chatbot
    """
    chatbot = VehicleMaintenanceChatbot()
    
    print("="*50)
    print("🤖 CHATBOT DỰ ĐOÁN BẢO DƯỠNG XE MÁY")
    print("="*50)
    print("Chatbot đã sẵn sàng! Nhập 'quit' để thoát.\n")
    
    while True:
        user_input = input("Bạn: ")
        
        if user_input.lower() in ['quit', 'thoát', 'exit', 'stop']:
            print("🤖 Chatbot: Tạm biệt! Hãy chăm sóc xe thật tốt nhé!")
            break
        
        response = chatbot.chat(user_input)
        print(f"\n🤖 Chatbot: {response}\n")

if __name__ == "__main__":
    demo_chatbot()