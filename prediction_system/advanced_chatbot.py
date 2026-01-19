import joblib
import pandas as pd
import json
import re
import speech_recognition as sr
import pyttsx3
import threading
import queue
from datetime import datetime

from openai import OpenAI

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

class AdvancedVehicleMaintenanceChatbot:
    def __init__(self, model_path="backend/model/rf_maintenance_new.pkl", api_key=None):
        """
        Khởi tạo chatbot nâng cao với LLM và mô hình RandomForest
        """
        self.model_path = model_path
        self.model = self.load_model()
        self.conversation_state = {}
        self.expected_inputs = [
            'total_km', 'avg_km_per_trip', 'trips_per_day', 'vehicle_age_months'
        ]

        # Khởi tạo LLM client
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None  # Sẽ sử dụng mô hình cục bộ nếu không có API key

        # Khởi tạo engine text-to-speech
        self.tts_engine = pyttsx3.init()
        self.setup_tts_voice()

        # Khởi tạo speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Queue cho voice processing
        self.voice_queue = queue.Queue()
        
    def setup_tts_voice(self):
        """
        Thiết lập giọng nói cho TTS
        """
        voices = self.tts_engine.getProperty('voices')
        for voice in voices:
            if "Vietnamese" in voice.name or "vietnam" in voice.name.lower():
                self.tts_engine.setProperty('voice', voice.id)
                break
        self.tts_engine.setProperty('rate', 150)  # Điều chỉnh tốc độ nói
    
    def load_model(self):
        try:
            model = joblib.load(self.model_path)
            return model
        except Exception as e:
            return None
    
    def preprocess_input(self, user_input):
        """
        Tiền xử lý đầu vào của người dùng
        """
        # Chuyển sang chữ thường và loại bỏ khoảng trắng dư
        user_input = user_input.lower().strip()

        # Trích xuất số từ chuỗi đầu vào (bao gồm cả số nguyên và số thập phân)
        # Biểu thức chính quy để bắt cả số nguyên và số thập phân (dùng dấu chấm hoặc dấu phẩy)
        numbers = re.findall(r'\d+(?:[,.]\d+)?', user_input)
        # Lọc bỏ các chuỗi không phải số hợp lệ và chuyển đổi dấu phẩy thành dấu chấm thập phân
        valid_numbers = []
        for num in numbers:
            # Thay dấu phẩy (,) bằng dấu chấm (.) để chuyển đổi thành số thập phân
            clean_num = num.replace(',', '.')
            try:
                # Kiểm tra xem có thể chuyển đổi thành số không
                float(clean_num)
                valid_numbers.append(clean_num)
            except ValueError:
                continue
        return valid_numbers
    
    def get_maintenance_status(self, total_km, avg_km_per_trip, trips_per_day, vehicle_age_months):
        """
        Dự đoán trạng thái bảo dưỡng bằng mô hình RandomForest phân loại
        """
        if self.model is None:
            return None, None

        try:
            # Chuyển đổi dữ liệu đầu vào sang kiểu số hợp lệ
            total_km = float(total_km)
            avg_km_per_trip = float(avg_km_per_trip)
            trips_per_day = int(trips_per_day)
            vehicle_age_months = int(vehicle_age_months)

            # Tạo DataFrame từ dữ liệu đầu vào
            input_data = pd.DataFrame([{
                "total_km": total_km,
                "avg_km_per_trip": avg_km_per_trip,
                "trips_per_day": trips_per_day,
                "vehicle_age_months": vehicle_age_months
            }])

            # Dự đoán (0 = không cần bảo dưỡng, 1 = cần bảo dưỡng)
            prediction = self.model.predict(input_data)[0]
            probability = self.model.predict_proba(input_data)[0] if hasattr(self.model, 'predict_proba') else None

            return prediction, probability
        except Exception as e:
            print(f"Lỗi khi dự đoán: {e}")
            return None, None
    
    def get_status_message(self, prediction):
        """
        Trả về thông báo trạng thái dựa trên kết quả dự đoán phân loại
        """
        if prediction == 1:
            return "WARNING: Car needs maintenance!"
        else:
            return "OK: Car is operating normally"
    
    def enhance_with_llm(self, user_input, ai_response):
        maintenance_keywords = ['bảo dưỡng', 'bảo duỡng', 'sửa chữa', 'hỏng', 'hỏng hóc', 'lốp', 'phanh', 'máy', 'động cơ', 'gara', 'dầu nhớt', 'thay nhớt', 'thay dầu', 'lọc gió', 'bugi', 'ắc quy', 'xăng', 'nhiên liệu', 'bình ắc quy', 'bugi', 'nhớt', 'dầu máy', 'dầu phanh', 'lọc nhớt', 'lọc xăng', 'lọc khí', 'nhông sên dĩa', 'côn', 'ly hợp', 'hộp số', 'mát xe', 'nhiệt độ', 'nổ máy', 'đề máy', 'khởi động', 'tiếng ồn', 'tiếng kêu', 'rung', 'giật', 'chập chờn', 'mất thắng', 'mất phanh', 'hỏng đèn', 'đèn không sáng', 'hao xăng', 'hao nhiên liệu', 'tiêu hao nhiên liệu']
        is_maintenance_related = any(keyword in user_input.lower() for keyword in maintenance_keywords)

        if is_maintenance_related:
            return self._generate_maintenance_advice(user_input, ai_response)
        else:
            return ai_response

    def _generate_maintenance_advice(self, user_input, ai_response):
        user_input_lower = user_input.lower()

        # Nhớt/Dầu nhớt
        if any(keyword in user_input_lower for keyword in ['nhớt', 'dầu nhớt', 'thay nhớt', 'dầu máy']):
            if any(q in user_input_lower for q in ['bao lâu', 'khi nào', 'sau bao nhiêu', 'thay bao lâu', 'lâu thay']):
                return f"Thông thường, bạn nên thay nhớt xe máy sau mỗi 1,000 - 3,000 km tùy theo loại nhớt và điều kiện sử dụng. Với nhớt tổng hợp cao cấp, có thể kéo dài đến 5,000 km. Tuy nhiên, nếu bạn sử dụng xe thường xuyên trong điều kiện khắc nghiệt (nhiều bụi, tải nặng, thời tiết xấu), nên thay nhớt sớm hơn định kỳ khuyến nghị."
            elif any(q in user_input_lower for q in ['loại nào tốt', 'nên dùng', 'dùng loại gì', 'chọn nhớt', 'nhớt gì']):
                return f"Đối với xe máy, bạn nên sử dụng nhớt có chỉ số API SG trở lên (tốt nhất là SH, SJ hoặc cao hơn) và độ nhớt phù hợp với khuyến cáo của nhà sản xuất (thường là 10W-30, 10W-40 hoặc 20W-50). Nhớt tổng hợp (synthetic) thường cho hiệu suất tốt hơn nhớt khoáng (mineral oil)."
            elif any(q in user_input_lower for q in ['dấu hiệu', 'biết', 'hết hạn', 'cần thay']):
                return f"Dấu hiệu cần thay nhớt bao gồm: màu nhớt chuyển sang đen sẫm, nhớt đặc quánh, có mùi khét, hoặc xe hoạt động ì̀n hơn bình thường. Ngoài ra, nếu đã đạt đến số km khuyến nghị thay nhớt, bạn nên thay dù nhớt vẫn còn trong tình trạng tốt."
            elif any(q in user_input_lower for q in ['cách thay', 'tự thay', 'thay ở nhà']):
                return f"Để tự thay nhớt tại nhà, bạn cần: 1) Nâng xe lên bằng chân chống giữa hoặc giá đỡ. 2) Tháo nắp nhớt và xả nhớt cũ ra. 3) Thay lọc nhớt nếu cần. 4) Xiết chặt ốc xả nhớt. 5) Đổ nhớt mới vào (theo dung tích khuyến cáo). 6) Kiểm tra mức nhớt qua que thăm dầu."
            else:
                return f"Việc thay nhớt định kỳ rất quan trọng để bảo vệ động cơ xe máy. Nhớt giúp bôi trơn, làm mát và làm sạch động cơ. Nếu không thay nhớt đúng hạn, động cơ có thể bị mài mòn nhanh chóng."

        # Lốp xe
        elif any(keyword in user_input_lower for keyword in ['lốp', 'vỏ', 'lốp xe', 'vỏ xe']):
            if any(q in user_input_lower for q in ['thay khi nào', 'bao lâu thay', 'khi nào thay', 'thay bao lâu']):
                return f"Bạn nên kiểm tra lốp xe định kỳ và thay khi độ mòn đạt đến vạch mòn (TWI) được đánh dấu trên lốp. Thông thường, lốp xe máy có thể sử dụng từ 8,000 - 15,000 km tùy theo loại lốp, điều kiện sử dụng và cách lái xe."
            elif any(q in user_input_lower for q in ['kiểm tra', 'cách kiểm', 'xem như thế nào']):
                return f"Để kiểm tra lốp, bạn nên quan sát độ mòn của họa tiết trên mặt lốp, kiểm tra áp suất lốp định kỳ (thường là 2.0 bar cho trước và 2.2 bar cho sau), và kiểm tra có vết nứt, đinh, hoặc vật sắc nhọn nào không."
            elif any(q in user_input_lower for q in ['áp suất', 'hơi', 'bơm bao nhiêu']):
                return f"Áp suất lốp tiêu chuẩn thường là 2.0 bar cho bánh trước và 2.2 bar cho bánh sau (có thể thay đổi theo từng loại xe). Bạn nên kiểm tra áp suất khi lốp nguội và bơm đúng áp suất khuyến cáo để đảm bảo độ bám và tuổi thọ lốp."
            else:
                return f"Lốp xe là bộ phận tiếp xúc trực tiếp với mặt đường, ảnh hưởng đến độ bám, an toàn và hiệu suất lái. Bạn nên kiểm tra áp suất lốp thường xuyên và thay lốp khi thấy dấu hiệu mòn bất thường hoặc hư hỏng."

        # Phanh
        elif any(keyword in user_input_lower for keyword in ['phanh', 'má phanh', 'dầu phanh', 'thắng', 'má thắng']):
            if any(q in user_input_lower for q in ['thay khi nào', 'kiểm tra', 'bảo dưỡng', 'thay bao lâu']):
                return f"Bạn nên kiểm tra má phanh mỗi 3,000 - 5,000 km và thay khi độ dày còn dưới 2mm. Dầu phanh nên được thay định kỳ 2 năm/lần hoặc theo khuyến cáo của nhà sản xuất để đảm bảo hiệu quả phanh."
            elif any(q in user_input_lower for q in ['yếu phanh', 'không ăn', 'ăn yếu']):
                return f"Nếu phanh yếu hoặc không ăn, có thể do má phanh mòn, dầu phanh hết hoặc có không khí trong hệ thống, hoặc đĩa phanh bị cong. Bạn nên kiểm tra và bảo dưỡng hệ thống phanh sớm để đảm bảo an toàn."
            elif any(q in user_input_lower for q in ['kêu', 'kêu cót', 'tiếng lạ']):
                return f"Tiếng kêu cót khi phanh có thể do má phanh mòn, bụi bẩn bám vào đĩa phanh, hoặc má phanh bị cứng. Nếu tiếng kêu lớn và liên tục, bạn nên kiểm tra hệ thống phanh ngay để đảm bảo an toàn."
            else:
                return f"Hệ thống phanh là yếu tố quan trọng cho an toàn khi lái xe. Bạn nên kiểm tra định kỳ má phanh, đĩa phanh và dầu phanh để đảm bảo hệ thống hoạt động hiệu quả."

        # Bugi
        elif any(keyword in user_input_lower for keyword in ['bugi', 'xiết', 'điện cực', 'đánh lửa']):
            if any(q in user_input_lower for q in ['thay khi nào', 'thay bao lâu', 'khi nào thay']):
                return f"Bugi nên được thay sau mỗi 10,000 - 15,000 km hoặc khi xuất hiện dấu hiệu đánh lửa kém, khó khởi động, hao xăng. Bugi iridium có thể kéo dài đến 30,000 km."
            elif any(q in user_input_lower for q in ['dấu hiệu', 'biết', 'hỏng', 'cần thay']):
                return f"Dấu hiệu bugi cần thay bao gồm: khó khởi động xe, máy hoạt động không ổn định, hao xăng bất thường, hoặc động cơ bị giật cục khi tăng tốc. Bạn có thể kiểm tra trực tiếp bugi để xem điện cực có bị mòn, bám carbon hay nứt vỡ không."
            elif any(q in user_input_lower for q in ['vệ sinh', 'làm sạch', 'làm mới']):
                return f"Để vệ sinh bugi, bạn có thể dùng bàn chải kim loại nhẹ nhàng làm sạch điện cực, loại bỏ muội than. Tuy nhiên, nếu bugi quá mòn hoặc hỏng, tốt nhất nên thay mới để đảm bảo hiệu suất động cơ."
            else:
                return f"Bugi đóng vai trò đánh lửa để đốt cháy hỗn hợp nhiên liệu - không khí trong buồng đốt. Bugi bị bẩn hoặc mòn sẽ làm giảm hiệu suất động cơ."

        # Ắc quy
        elif any(keyword in user_input_lower for keyword in ['ắc quy', 'ac quy', 'ắcquy', 'battery', 'pin']):
            if any(q in user_input_lower for q in ['thay khi nào', 'tuổi thọ', 'bao lâu']):
                return f"Ắc quy xe máy thường có tuổi thọ từ 2 - 3 năm. Bạn nên thay khi thấy dấu hiệu khó đề, đèn yếu, hoặc sau 2 năm sử dụng nếu thường xuyên sử dụng đèn, còi nhiều."
            elif any(q in user_input_lower for q in ['kiểm tra', 'test', 'test ac quy']):
                return f"Để kiểm tra ắc quy, bạn có thể dùng đồng hồ vạn năng đo điện áp: trên 12.6V là tốt, từ 12.0-12.4V là trung bình, dưới 12.0V là yếu. Ngoài ra, kiểm tra các đầu cực có bị oxi hóa không và đảm bảo ắc quy được sạc đầy đủ."
            elif any(q in user_input_lower for q in ['vệ sinh', 'làm sạch', 'chăm sóc']):
                return f"Để chăm sóc ắc quy, bạn nên thường xuyên kiểm tra và làm sạch các đầu cực bằng baking soda và nước. Đảm bảo nắp ắc quy kín và không để ắc quy bị xả hết. Nếu xe không sử dụng lâu ngày, nên tháo ắc quy hoặc sạc định kỳ."
            else:
                return f"Ắc quy cung cấp điện cho hệ thống đánh lửa, đèn và các thiết bị điện khác. Bạn nên kiểm tra định kỳ và đảm bảo ắc quy luôn được sạc đầy đủ."

        # Lọc gió
        elif any(keyword in user_input_lower for keyword in ['lọc gió', 'bầu lọc gió', 'filter', 'khí']):
            if any(q in user_input_lower for q in ['thay khi nào', 'thay bao lâu', 'khi nào thay']):
                return f"Lọc gió nên được thay sau mỗi 6,000 - 10,000 km hoặc sớm hơn nếu bạn thường xuyên đi đường bụi. Lọc gió sạch giúp động cơ hoạt động hiệu quả và tiết kiệm nhiên liệu."
            elif any(q in user_input_lower for q in ['làm sạch', 'vệ sinh', 'rửa']):
                return f"Lọc gió giấy không nên rửa mà chỉ nên thổi nhẹ bụi bằng khí nén. Lọc gió bằng dầu (oil filter) có thể được làm sạch và tra dầu lại. Tuy nhiên, nếu lọc quá bẩn, tốt nhất nên thay mới."
            elif any(q in user_input_lower for q in ['tắc', 'bẩn', 'ảnh hưởng']):
                return f"Lọc gió bẩn sẽ làm giảm lượng không khí vào động cơ, gây hao xăng, giảm công suất và có thể làm hỏng bugi. Nếu bạn thấy xe yếu, hao xăng bất thường, có thể do lọc gió bị tắc."
            else:
                return f"Lọc gió giúp ngăn bụi và tạp chất vào động cơ, bảo vệ động cơ khỏi mài mòn. Lọc gió bẩn sẽ làm giảm hiệu suất động cơ và tăng tiêu hao nhiên liệu."

        # Máy/Động cơ
        elif any(keyword in user_input_lower for keyword in ['máy', 'động cơ', 'nổ máy', 'đề máy', 'khởi động', 'nhiệt độ']):
            if any(q in user_input_lower for q in ['nóng', 'nhiệt độ cao', 'quá nhiệt']):
                return f"Động cơ nóng quá mức có thể do thiếu nước làm mát (xe có làm mát bằng nước), nhớt kém chất lượng, tắc ống xả khí nóng, hoặc tải trọng quá mức. Bạn nên dừng xe nghỉ ngơi và kiểm tra hệ thống làm mát."
            elif any(q in user_input_lower for q in ['khó nổ', 'khó đề', 'không nổ', 'không đề']):
                return f"Nguyên nhân khó nổ máy có thể do ắc quy yếu, bugi hỏng, lọc gió tắc, hoặc hệ thống nhiên liệu có vấn đề. Bạn nên kiểm tra từng bộ phận theo thứ tự: ắc quy, bugi, lọc gió, và cuối cùng là hệ thống nhiên liệu."
            elif any(q in user_input_lower for q in ['kêu', 'ồn', 'tiếng lạ', 'gằn']):
                return f"Tiếng kêu lạ từ động cơ có thể do thiếu nhớt, bugi hỏng, hoặc các bộ phận cơ khí bị mòn. Nếu tiếng kêu lớn hoặc liên tục, bạn nên đưa xe đến trung tâm bảo dưỡng để kiểm tra kỹ hơn."
            else:
                return f"Động cơ là trái tim của xe máy. Bạn nên bảo dưỡng định kỳ, thay nhớt đúng hạn, và kiểm tra các hệ thống liên quan để đảm bảo động cơ hoạt động hiệu quả và bền bỉ."

        # Xăng/Nhiên liệu
        elif any(keyword in user_input_lower for keyword in ['xăng', 'nhiên liệu', 'chai xăng', 'bình xăng', 'hao xăng', 'tiêu hao nhiên liệu']):
            if any(q in user_input_lower for q in ['hao xăng', 'tiêu hao', 'ăn xăng']):
                return f"Xe hao xăng có thể do nhiều nguyên nhân: lọc gió bẩn, bugi hỏng, áp suất lốp không đúng, tải trọng quá mức, hoặc lái xe không đều. Bạn nên kiểm tra và bảo dưỡng các bộ phận liên quan để cải thiện mức tiêu hao nhiên liệu."
            elif any(q in user_input_lower for q in ['chứa bao nhiêu', 'dung tích', 'bình xăng bao nhiêu']):
                return f"Dung tích bình xăng xe máy thường dao động từ 4-8 lít tùy theo loại xe. Bạn có thể kiểm tra trong sổ tay hướng dẫn sử dụng của xe để biết chính xác dung tích bình xăng của mẫu xe cụ thể."
            elif any(q in user_input_lower for q in ['loại nào', 'dùng xăng gì', 'nên dùng']):
                return f"Đa số xe máy hiện nay sử dụng xăng RON 92 hoặc 95. Bạn nên dùng loại xăng theo khuyến cáo của nhà sản xuất (thường là RON 92). Sử dụng xăng kém chất lượng có thể làm giảm hiệu suất và gây hại cho động cơ."
            else:
                return f"Hệ thống nhiên liệu cần được bảo dưỡng định kỳ để đảm bảo hiệu suất động cơ. Bạn nên sử dụng xăng đúng loại (RON 92 hoặc 95), và kiểm tra hệ thống nhiên liệu định kỳ."

        # Tổng quát về bảo dưỡng
        elif any(keyword in user_input_lower for keyword in ['bảo dưỡng', 'bảo trì', 'định kỳ', 'lịch bảo dưỡng', 'khi nào bảo dưỡng']):
            return f"Bảo dưỡng xe máy định kỳ rất quan trọng để duy trì hiệu suất và độ bền của xe. Lịch bảo dưỡng cơ bản bao gồm: thay nhớt (1,000-3,000km), kiểm tra bugi (10,000km), thay lọc gió (6,000-10,000km), kiểm tra má phanh (3,000-5,000km), và kiểm tra ắc quy (6 tháng/lần)."

        # Bảo quản xe
        elif any(keyword in user_input_lower for keyword in ['bảo quản', 'cất xe', 'để lâu', 'bảo quản lâu dài']):
            if any(q in user_input_lower for q in ['để lâu', 'không đi', 'cất đi', 'bảo quản lâu']):
                return f"Khi bảo quản xe lâu ngày, bạn nên: 1) Làm sạch xe kỹ lưỡng. 2) Thay nhớt và lọc gió mới. 3) Bơm căng lốp. 4) Cho thêm phụ gia nhiên liệu nếu để trên 30 ngày. 5) Dùng bạt che mưa nắng. 6) Nên nổ máy định kỳ 1 tuần/lần khoảng 5-10 phút."
            else:
                return f"Để bảo quản xe tốt, bạn nên thường xuyên lau chùi, kiểm tra các bộ phận, và để xe nơi khô ráo. Tránh để xe ngoài trời mưa nắng lâu ngày. Nên sử dụng bạt phủ xe để bảo vệ sơn và các bộ phận nhựa."

        # Vệ sinh xe
        elif any(keyword in user_input_lower for keyword in ['rửa xe', 'vệ sinh', 'làm sạch', 'chăm sóc ngoại thất']):
            if any(q in user_input_lower for q in ['cách rửa', 'rửa như thế nào', 'bước nào trước']):
                return f"Để rửa xe đúng cách: 1) Rửa sơ bằng nước để loại bỏ bụi bẩn. 2) Dùng xà phòng chuyên dụng và khăn mềm rửa từ trên xuống dưới. 3) Rửa kỹ phần gầm máy và bánh xe. 4) Tráng sạch xà phòng. 5) Lau khô bằng khăn mềm. 6) Bôi trơn các phần chuyển động nếu cần."
            else:
                return f"Vệ sinh xe định kỳ giúp bảo vệ lớp sơn và các bộ phận kim loại khỏi gỉ sét. Nên rửa xe mỗi 1-2 tuần và sử dụng các sản phẩm chăm sóc chuyên dụng để bảo vệ bề mặt xe."

        # Lốp đặc/hơi
        elif any(keyword in user_input_lower for keyword in ['lốp đặc', 'lốp hơi', 'so sánh', 'loại nào tốt']):
            return f"Lốp đặc không cần bơm hơi và ít bị xì, nhưng độ êm thấp hơn. Lốp hơi có độ êm tốt hơn, bám đường tốt hơn, nhưng cần kiểm tra áp suất thường xuyên và có thể bị xì. Tùy theo mục đích sử dụng mà chọn loại phù hợp."

        # Dầu phanh
        elif any(keyword in user_input_lower for keyword in ['dầu phanh', 'thay dầu phanh', 'chảy', 'bị chảy']):
            if any(q in user_input_lower for q in ['thay khi nào', 'định kỳ', 'bao lâu']):
                return f"Dầu phanh nên được thay định kỳ 2 năm/lần hoặc theo khuyến cáo của nhà sản xuất. Dầu phanh hút ẩm theo thời gian, làm giảm hiệu quả phanh. Nếu thấy phanh yếu hoặc có hiện tượng bất thường, nên kiểm tra và thay dầu phanh ngay."
            else:
                return f"Dầu phanh rất quan trọng cho hệ thống phanh hoạt động hiệu quả. Dầu phanh cần được thay định kỳ để đảm bảo an toàn khi lái xe."

        # Dầu máy
        elif any(keyword in user_input_lower for keyword in ['dầu máy', 'nhớt máy', 'dầu động cơ']):
            if any(q in user_input_lower for q in ['loại nào', 'chọn như thế nào', 'chênh lệch']):
                return f"Dầu máy (nhớt máy) có ba loại chính: khoáng (mineral), bán tổng hợp (semi-synthetic), và tổng hợp (fully synthetic). Dầu tổng hợp có hiệu suất tốt nhất nhưng giá cao hơn. Bạn nên chọn loại phù hợp với khuyến cáo của nhà sản xuất và điều kiện sử dụng."
            else:
                return f"Dầu máy giúp bôi trơn, làm mát và làm sạch động cơ. Việc thay dầu máy định kỳ rất quan trọng để bảo vệ động cơ khỏi mài mòn."

        # Ly hợp/côn
        elif any(keyword in user_input_lower for keyword in ['ly hợp', 'côn', 'đề pa', 'cháy côn']):
            if any(q in user_input_lower for q in ['cháy côn', 'hở côn', 'kéo côn']):
                return f"Dấu hiệu cháy côn bao gồm: xe yếu, tăng tốc kém dù ga lớn, mùi khét từ ly hợp. Nguyên nhân thường do kéo côn, đề pa liên tục, hoặc côn bị mòn. Nếu bị cháy côn, nên đưa xe đến garage để kiểm tra và thay thế nếu cần."
            else:
                return f"Ly hợp (côn) giúp truyền lực từ động cơ đến bánh xe. Bạn nên sử dụng côn đúng cách để tránh bị mòn sớm và tăng tuổi thọ cho hệ thống truyền động."

        # Hệ thống điện
        elif any(keyword in user_input_lower for keyword in ['đèn', 'điện', 'còi', 'hệ thống điện', 'bóng đèn']):
            if any(q in user_input_lower for q in ['thay bóng', 'đổi đèn', 'độ đèn']):
                return f"Khi thay bóng đèn, bạn nên chọn loại có công suất và kích thước phù hợp với xe. Không nên độ đèn công suất cao hơn khuyến cáo vì có thể gây quá tải cho hệ thống điện và làm giảm tuổi thọ các thiết bị khác."
            else:
                return f"Hệ thống điện bao gồm đèn, còi, và các thiết bị điện khác. Bạn nên kiểm tra định kỳ các tiếp điểm, dây dẫn và thay thế các thiết bị hỏng để đảm bảo an toàn khi lái xe."

        else:
            return ai_response
    
    def generate_response(self, user_input):
        """
        Sinh phản hồi dựa trên đầu vào của người dùng
        """
        user_input_lower = user_input.lower()

        # Phản hồi chào mừng
        if any(greeting in user_input_lower for greeting in ['xin chào', 'chào', 'hello', 'hi', 'chào buổi sáng', 'chào buổi chiều', 'chào buổi tối']):
            base_response = ("🤖 Xin chào! Tôi là Chatbot dự đoán bảo dưỡng xe máy thông minh.\n\n"
                           "Tôi có thể giúp bạn:\n"
                           "• Dự đoán thời điểm cần bảo dưỡng xe\n"
                           "• Kiểm tra tình trạng xe\n"
                           "• Trả lời các câu hỏi về bảo dưỡng xe\n\n"
                           "Vui lòng cung cấp các thông tin sau:\n"
                           "1. Tổng km đã chạy\n"
                           "2. Km trung bình mỗi chuyến\n"
                           "3. Số chuyến/ngày\n"
                           "4. Tuổi xe (tháng)")
            return self.enhance_with_llm(user_input, base_response)

        # Phản hồi thông tin hỗ trợ
        elif any(info in user_input_lower for info in ['giúp', 'hướng dẫn', 'trợ giúp', 'help', 'cách dùng', 'sử dụng']):
            base_response = ("ℹ️ Tôi có thể giúp bạn dự đoán thời điểm cần bảo dưỡng xe máy.\n\n"
                           "Vui lòng cung cấp các thông tin sau:\n"
                           "1. Tổng km đã chạy\n"
                           "2. Km trung bình mỗi chuyến\n"
                           "3. Số chuyến/ngày\n"
                           "4. Tuổi xe (tháng)\n\n"
                           "Bạn có thể nói theo cách tự nhiên, ví dụ:\n"
                           "'Tôi đã chạy 15000 km, trung bình mỗi chuyến 20 km, đi 4 chuyến mỗi ngày, xe đã 12 tháng tuổi'")
            return self.enhance_with_llm(user_input, base_response)

        # Kiểm tra xem người dùng có cung cấp đủ thông tin không
        numbers = self.preprocess_input(user_input)

        if len(numbers) >= 4:
            try:
                # Lấy 4 số đầu tiên theo thứ tự mong muốn
                total_km = float(numbers[0])
                avg_km_per_trip = float(numbers[1])
                trips_per_day = int(float(numbers[2]))
                vehicle_age_months = int(float(numbers[3]))

                # Dự đoán trạng thái bảo dưỡng
                prediction, probability = self.get_maintenance_status(
                    total_km, avg_km_per_trip, trips_per_day, vehicle_age_months
                )

                if prediction is not None:
                    status_msg = self.get_status_message(prediction)

                    # Tạo phản hồi cơ bản
                    base_response = (
                        f"🔍 **KẾT QUẢ DỰ ĐOÁN**\n\n"
                        f"📊 Thông tin xe:\n"
                        f"- Tổng km đã chạy: {total_km} km\n"
                        f"- Km trung bình mỗi chuyến: {avg_km_per_trip} km\n"
                        f"- Số chuyến/ngày: {trips_per_day}\n"
                        f"- Tuổi xe: {vehicle_age_months} tháng\n\n"
                        f"🔔 Trạng thái: {status_msg}\n\n"
                    )

                    # Thêm thông tin xác suất nếu có
                    if probability is not None:
                        prob_no_maintenance = probability[0]  # Xác suất không cần bảo dưỡng
                        prob_need_maintenance = probability[1]  # Xác suất cần bảo dưỡng
                        base_response += f"📊 Xác suất:\n"
                        base_response += f"- Không cần bảo dưỡng: {prob_no_maintenance:.2%}\n"
                        base_response += f"- Cần bảo dưỡng: {prob_need_maintenance:.2%}\n"

                    return self.enhance_with_llm(user_input, base_response)
                else:
                    base_response = "❌ Rất tiếc, không thể thực hiện dự đoán. Vui lòng kiểm tra lại mô hình."
                    return self.enhance_with_llm(user_input, base_response)

            except ValueError:
                base_response = ("❌ Dữ liệu không hợp lệ. Vui lòng cung cấp:\n"
                               "1. Tổng km đã chạy\n"
                               "2. Km trung bình mỗi chuyến\n"
                               "3. Số chuyến/ngày\n"
                               "4. Tuổi xe (tháng)")
                return self.enhance_with_llm(user_input, base_response)

        # Nếu không có đủ thông tin, kiểm tra xem có phải là câu hỏi chung về xe không
        else:
            # Kiểm tra nếu người dùng hỏi về các vấn đề chung liên quan đến xe
            maintenance_related = any(keyword in user_input_lower for keyword in ['bảo dưỡng', 'sửa chữa', 'hỏng', 'hỏng hóc', 'lốp', 'phanh', 'máy', 'động cơ', 'gara', 'dầu nhớt', 'thay nhớt', 'thay dầu', 'lọc gió', 'bugi', 'ắc quy', 'xăng', 'nhiên liệu', 'bình ắc quy', 'nhớt', 'dầu máy', 'dầu phanh', 'lọc nhớt', 'lọc xăng', 'lọc khí', 'nhông sên dĩa', 'côn', 'ly hợp', 'hộp số', 'mát xe', 'nhiệt độ', 'nổ máy', 'đề máy', 'khởi động', 'tiếng ồn', 'tiếng kêu', 'rung', 'giật', 'chập chờn', 'mất thắng', 'mất phanh', 'hỏng đèn', 'đèn không sáng', 'hao xăng', 'hao nhiên liệu', 'tiêu hao nhiên liệu'])

            if maintenance_related:
                # Nếu là câu hỏi về bảo dưỡng, trả về phản hồi cơ bản và để LLM xử lý
                base_response = f"Đây là câu hỏi về bảo dưỡng xe: {user_input}"
                return self.enhance_with_llm(user_input, base_response)
            else:
                # Nếu không phải là câu hỏi về bảo dưỡng, yêu cầu người dùng cung cấp thông tin
                base_response = ("🤔 Vui lòng cung cấp đầy đủ thông tin:\n"
                               "• Tổng km đã chạy\n"
                               "• Km trung bình mỗi chuyến\n"
                               "• Số chuyến/ngày\n"
                               "• Tuổi xe (tháng)\n\n"
                               "Ví dụ: Tôi đã chạy 15000 km, trung bình mỗi chuyến 20 km, đi 4 chuyến mỗi ngày, xe đã 12 tháng tuổi\n\n"
                               "Hoặc bạn có thể hỏi tôi về các vấn đề bảo dưỡng xe, sửa chữa, hoặc các lỗi thường gặp trên xe máy.")
                return self.enhance_with_llm(user_input, base_response)
    
    def chat(self, user_input):
        """
        Hàm chính để trò chuyện với chatbot
        """
        if not user_input.strip():
            return "🤖 Xin vui lòng nhập câu hỏi hoặc thông tin của bạn."
        
        response = self.generate_response(user_input)
        return response
    
    def speak_text(self, text):
        """
        Phát âm văn bản
        """
        def speak_worker():
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        
        # Chạy trong thread riêng để không chặn luồng chính
        thread = threading.Thread(target=speak_worker)
        thread.start()
        thread.join()
    
    def listen_voice(self):
        """
        Nghe và nhận diện giọng nói
        """
        try:
            with self.microphone as source:
                print("Đang lắng nghe...")
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, timeout=5)
            
            print("Đang nhận diện...")
            # Sử dụng Google Speech Recognition
            text = self.recognizer.recognize_google(audio, language="vi-VN")
            return text
        except sr.WaitTimeoutError:
            return "Không nghe thấy âm thanh"
        except sr.UnknownValueError:
            return "Không thể nhận diện giọng nói"
        except sr.RequestError as e:
            return f"Lỗi kết nối dịch vụ nhận diện: {e}"
    
    def voice_chat(self):
        """
        Chế độ chat bằng giọng nói
        """
        self.speak_text("Xin chào! Tôi là chatbot dự đoán bảo dưỡng xe máy. Vui lòng nói thông tin xe của bạn.")
        
        while True:
            user_speech = self.listen_voice()
            print(f"Bạn nói: {user_speech}")
            
            if "thoát" in user_speech.lower() or "dừng" in user_speech.lower() or "tạm biệt" in user_speech.lower():
                self.speak_text("Tạm biệt! Hãy chăm sóc xe thật tốt nhé!")
                break
            
            if user_speech and user_speech != "Không nghe thấy âm thanh" and user_speech != "Không thể nhận diện giọng nói":
                response = self.chat(user_speech)
                print(f"Chatbot: {response}")
                self.speak_text(response.replace("**", "").replace("\n", ". "))
            else:
                self.speak_text("Tôi không nghe rõ, vui lòng nói lại.")

# Hàm demo
def demo_advanced_chatbot():
    """
    Hàm demo để thử nghiệm chatbot nâng cao
    """
    # Nếu bạn có OpenAI API key, hãy thay thế vào đây
    api_key = None  # Thay bằng API key của bạn nếu có
    
    chatbot = AdvancedVehicleMaintenanceChatbot(api_key=api_key)
    
    print("="*60)
    print("🤖 CHATBOT DỰ ĐOÁN BẢO DƯỠNG XE MÁY NÂNG CAO")
    print("Kết hợp LLM + RandomForest + Voice Chatbot")
    print("="*60)
    print("Chọn chế độ:")
    print("1. Chat văn bản")
    print("2. Chat bằng giọng nói")
    print("Nhập 'quit' để thoát.\n")
    
    choice = input("Chọn chế độ (1 hoặc 2): ")
    
    if choice == "2":
        print("Chuyển sang chế độ voice chatbot...")
        chatbot.voice_chat()
    elif choice == "1":
        print("Chatbot đã sẵn sàng! Nhập 'quit' để thoát.\n")
        
        while True:
            user_input = input("Bạn: ")
            
            if user_input.lower() in ['quit', 'thoát', 'exit', 'stop']:
                print("🤖 Chatbot: Tạm biệt! Hãy chăm sóc xe thật tốt nhé!")
                break
            
            response = chatbot.chat(user_input)
            print(f"\n🤖 Chatbot: {response}\n")
    else:
        print("Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    demo_advanced_chatbot()