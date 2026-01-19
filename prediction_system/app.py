from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd
import os
import json
import re
from datetime import datetime


app = Flask(__name__)

import os


MODEL_PATH = "maintenance_model.pkl"

def load_model():
    """Hàm tải mô hình từ file output sau khi train"""
    import os

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


    classification_new_path = os.path.join(project_root, "backend", "model", "rf_maintenance_new.pkl")
    regression_new_path = os.path.join(project_root, "maintenance_model_new.pkl")
    classification_old_path = os.path.join(project_root, "backend", "model", "rf_maintenance.pkl")
    regression_old_path = os.path.join(project_root, "maintenance_model.pkl")

    if os.path.exists(classification_new_path):
        model_path = classification_new_path
        print("Sử dụng mô hình phân loại mới (RandomForestClassifier) - dữ liệu hợp lý")
    elif os.path.exists(regression_new_path):
        model_path = regression_new_path
        print("Sử dụng mô hình hồi quy mới (RandomForestRegressor) - dữ liệu hợp lý")
    elif os.path.exists(classification_old_path):
        model_path = classification_old_path
        print("Sử dụng mô hình phân loại cũ (RandomForestClassifier)")
    elif os.path.exists(regression_old_path):
        model_path = regression_old_path
        print("Sử dụng mô hình hồi quy cũ (RandomForestRegressor)")
    else:
        print(f"Không tìm thấy file mô hình nào:")
        print(f"  - {classification_new_path}")
        print(f"  - {regression_new_path}")
        print(f"  - {classification_old_path}")
        print(f"  - {regression_old_path}")
        print("Vui lòng đảm bảo đã chạy quá trình huấn luyện để tạo file mô hình.")
        return None

    try:
        model = joblib.load(model_path)
        print(f"Đã tải mô hình thành công từ: {model_path}")
        print(f"Kích thước file mô hình: {os.path.getsize(model_path)} bytes")
        return model
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return None

# Load model
model = load_model()


from advanced_chatbot import AdvancedVehicleMaintenanceChatbot


chatbot = AdvancedVehicleMaintenanceChatbot()


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hệ Thống Dự Đoán Bảo Dưỡng Xe</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container {
            width: 100%;
            max-width: 800px;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
            position: relative;
        }

        .header {
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.2rem;
            margin-bottom: 10px;
        }

        .header p {
            opacity: 0.9;
            font-size: 1.1rem;
        }

        .content {
            padding: 40px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .form-group {
            margin-bottom: 20px;
        }

        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
            font-size: 1rem;
        }

        .form-group input {
            width: 100%;
            padding: 15px;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            font-size: 1rem;
            transition: all 0.3s ease;
        }

        .form-group input:focus {
            outline: none;
            border-color: #4CAF50;
            box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.1);
        }

        .btn-submit {
            width: 100%;
            padding: 16px;
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 10px;
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(76, 175, 80, 0.3);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        .result {
            margin-top: 30px;
            padding: 25px;
            border-radius: 8px;
            display: none;
            animation: fadeIn 0.5s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }

        .warning {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
        }

        .error {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }

        .result h3 {
            margin-bottom: 15px;
            font-size: 1.3rem;
        }

        .result strong {
            font-size: 1.2rem;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #4CAF50;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .stats-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            display: none;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }

        .stat-item {
            text-align: center;
            padding: 15px;
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
        }

        .stat-value {
            font-size: 1.8rem;
            font-weight: bold;
            margin-bottom: 5px;
        }

        .stat-label {
            font-size: 0.9rem;
            opacity: 0.9;
        }

        /* Chatbot Styles */
        .chatbot-container {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 1000;
        }

        .chatbot-button {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            color: white;
            border: none;
            cursor: pointer;
            font-size: 24px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
        }

        .chatbot-button:hover {
            transform: scale(1.1);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }

        .chatbot-window {
            position: absolute;
            bottom: 80px;
            right: 0;
            width: 350px;
            height: 500px;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            display: none;
            flex-direction: column;
            overflow: hidden;
        }

        .chatbot-header {
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            color: white;
            padding: 15px;
            text-align: center;
            font-weight: bold;
        }

        .chatbot-messages {
            flex: 1;
            padding: 15px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 10px;
            background: #f9f9f9;
        }

        .message {
            max-width: 80%;
            padding: 10px 15px;
            border-radius: 18px;
            line-height: 1.4;
            position: relative;
        }

        .user-message {
            background: #4CAF50;
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 5px;
        }

        .bot-message {
            background: #e9ecef;
            color: #333;
            align-self: flex-start;
            border-bottom-left-radius: 5px;
        }

        .chatbot-input {
            display: flex;
            padding: 15px;
            background: white;
            border-top: 1px solid #eee;
        }

        .chatbot-input input {
            flex: 1;
            padding: 12px 15px;
            border: 2px solid #e1e5e9;
            border-radius: 25px;
            font-size: 1rem;
            outline: none;
        }

        .chatbot-input input:focus {
            border-color: #4CAF50;
        }

        .chatbot-input button {
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 50%;
            width: 45px;
            height: 45px;
            margin-left: 10px;
            cursor: pointer;
            font-size: 1.2rem;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .chatbot-input button:hover {
            background: #45a049;
        }

        .typing-indicator {
            display: none;
            align-self: flex-start;
            background: #e9ecef;
            color: #666;
            padding: 10px 15px;
            border-radius: 18px;
            font-style: italic;
        }

        @media (max-width: 768px) {
            .content {
                padding: 20px;
            }

            .header {
                padding: 20px;
            }

            .header h1 {
                font-size: 1.8rem;
            }

            .form-grid {
                grid-template-columns: 1fr;
            }

            .chatbot-window {
                width: 300px;
                height: 400px;
                bottom: 70px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-car"></i> HỆ THỐNG DỰ ĐOÁN BẢO DƯỠNG XE</h1>
            <p>Nhập thông tin xe để nhận dự đoán thông minh từ AI</p>
        </div>

        <div class="content">
            <form id="predictionForm">
                <div class="form-grid">
                    <div class="form-group">
                        <label for="total_km"><i class="fas fa-road"></i> Tổng km đã chạy</label>
                        <input type="number" id="total_km" name="total_km" step="any" placeholder="Nhập tổng km..." required>
                    </div>

                    <div class="form-group">
                        <label for="avg_km_per_trip"><i class="fas fa-route"></i> Km trung bình mỗi chuyến</label>
                        <input type="number" id="avg_km_per_trip" name="avg_km_per_trip" step="any" placeholder="Nhập km trung bình..." required>
                    </div>

                    <div class="form-group">
                        <label for="trips_per_day"><i class="fas fa-calendar-alt"></i> Số chuyến/ngày</label>
                        <input type="number" id="trips_per_day" name="trips_per_day" placeholder="Nhập số chuyến..." required>
                    </div>

                    <div class="form-group">
                        <label for="vehicle_age_months"><i class="fas fa-clock"></i> Tuổi xe (tháng)</label>
                        <input type="number" id="vehicle_age_months" name="vehicle_age_months" placeholder="Nhập tuổi xe..." required>
                    </div>
                </div>

                <button type="submit" class="btn-submit">
                    <i class="fas fa-calculator"></i> Dự đoán Ngay
                </button>
            </form>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Đang xử lý dữ liệu với AI...</p>
            </div>

            <div class="stats-card" id="statsCard">
                <h3><i class="fas fa-chart-line"></i> Thống kê Dự Đoán</h3>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value" id="remainingKm">0</div>
                        <div class="stat-label">Còn lại (km)</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="statusText">Bình thường</div>
                        <div class="stat-label">Trạng thái</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="nextService">-</div>
                        <div class="stat-label">Bảo dưỡng sau</div>
                    </div>
                </div>
            </div>

            <div id="result" class="result"></div>
        </div>
    </div>

    <!-- Chatbot for maintenance advice -->
    <div class="chatbot-container">
        <button class="chatbot-button" id="chatbotToggle">
            <i class="fas fa-robot"></i>
        </button>
        <div class="chatbot-window" id="chatbotWindow">
            <div class="chatbot-header">
                <i class="fas fa-comments"></i> Chatbot Bảo Dưỡng Xe
            </div>
            <div class="chatbot-messages" id="chatbotMessages">
                <div class="message bot-message">
                    Xin chào! Tôi là chatbot chuyên về bảo trì bảo dưỡng xe máy.<br>
                    Bạn có thể hỏi tôi về:<br>
                    • Dấu hiệu nhận biết hỏng hóc<br>
                    • Thời điểm bảo dưỡng<br>
                    • Chi phí sửa chữa ước lượng<br>
                    • Cách kiểm tra các bộ phận
                </div>
            </div>
            <div class="typing-indicator" id="typingIndicator">
                <i class="fas fa-circle-notch fa-spin"></i> Chatbot đang suy nghĩ...
            </div>
            <div class="chatbot-input">
                <input type="text" id="chatbotInput" placeholder="Hỏi về bảo dưỡng xe...">
                <button id="chatbotSend"><i class="fas fa-paper-plane"></i></button>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();

            // Hiển thị trạng thái đang xử lý
            const loadingDiv = document.getElementById('loading');
            const resultDiv = document.getElementById('result');
            const statsCard = document.getElementById('statsCard');

            loadingDiv.style.display = 'block';
            resultDiv.style.display = 'none';
            statsCard.style.display = 'none';

            // Lấy dữ liệu từ form
            const formData = {
                total_km: parseFloat(document.getElementById('total_km').value),
                avg_km_per_trip: parseFloat(document.getElementById('avg_km_per_trip').value),
                trips_per_day: parseInt(document.getElementById('trips_per_day').value),
                vehicle_age_months: parseInt(document.getElementById('vehicle_age_months').value)
            };

            try {
                // Gửi yêu cầu POST đến backend Flask
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(formData)
                });

                const data = await response.json();

                // Ẩn trạng thái đang xử lý
                loadingDiv.style.display = 'none';

                if (response.ok) {
                    // Hiển thị kết quả
                    let message = `<h3><i class="fas fa-robot"></i> KẾT QUẢ AI</h3>`;
                    message += `<p>Dự đoán còn khoảng <strong>${Math.round(data.remaining_km)}</strong> km đến kỳ bảo dưỡng</p>`;

                    if (data.remaining_km <= 500) {
                        message += `<p><i class="fas fa-exclamation-triangle"></i> <strong>⚠️ CẢNH BÁO: Xe sắp cần bảo dưỡng!</strong></p>`;
                        resultDiv.className = 'result warning';

                        // Cập nhật thống kê
                        document.getElementById('statusText').textContent = 'Cảnh báo';
                        document.getElementById('statusText').style.color = '#856404';
                    } else {
                        message += `<p><i class="fas fa-check-circle"></i> <strong>✅ Xe vẫn đang hoạt động bình thường</strong></p>`;
                        resultDiv.className = 'result success';

                        // Cập nhật thống kê
                        document.getElementById('statusText').textContent = 'Bình thường';
                        document.getElementById('statusText').style.color = 'white';
                    }

                    resultDiv.innerHTML = message;
                    resultDiv.style.display = 'block';

                    // Cập nhật thẻ thống kê
                    document.getElementById('remainingKm').textContent = Math.round(data.remaining_km);
                    document.getElementById('nextService').textContent = data.remaining_km <= 500 ? 'Ngay lập tức' : 'Sau ' + Math.round(data.remaining_km) + ' km';
                    statsCard.style.display = 'block';

                } else {
                    resultDiv.innerHTML = `<p><i class="fas fa-exclamation-circle"></i> Lỗi: ${data.error || 'Có lỗi xảy ra khi xử lý yêu cầu'}</p>`;
                    resultDiv.className = 'result error';
                    resultDiv.style.display = 'block';
                }
            } catch (error) {
                // Ẩn trạng thái đang xử lý
                loadingDiv.style.display = 'none';

                resultDiv.innerHTML = `<p><i class="fas fa-plug"></i> Lỗi kết nối: ${error.message}</p>`;
                resultDiv.className = 'result error';
                resultDiv.style.display = 'block';
            }
        });

        // Chatbot functionality
        const chatbotToggle = document.getElementById('chatbotToggle');
        const chatbotWindow = document.getElementById('chatbotWindow');
        const chatbotMessages = document.getElementById('chatbotMessages');
        const chatbotInput = document.getElementById('chatbotInput');
        const chatbotSend = document.getElementById('chatbotSend');
        const typingIndicator = document.getElementById('typingIndicator');

        // Toggle chatbot window
        chatbotToggle.addEventListener('click', function() {
            const isVisible = chatbotWindow.style.display === 'flex';
            chatbotWindow.style.display = isVisible ? 'none' : 'flex';
        });

        // Send message when clicking send button
        chatbotSend.addEventListener('click', sendMessage);

        // Send message when pressing Enter
        chatbotInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        function sendMessage() {
            const message = chatbotInput.value.trim();
            if (!message) return;

            // Add user message
            addMessage(message, 'user');
            chatbotInput.value = '';

            // Show typing indicator
            typingIndicator.style.display = 'block';
            scrollToBottom();

            // Send request to backend
            fetch('/chat_maintenance', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: message })
            })
            .then(response => response.json())
            .then(data => {
                // Hide typing indicator
                typingIndicator.style.display = 'none';

                // Add bot response
                addMessage(data.response, 'bot');
            })
            .catch(error => {
                // Hide typing indicator
                typingIndicator.style.display = 'none';

                // Add error message
                addMessage('Xin lỗi, đã xảy ra lỗi khi xử lý yêu cầu.', 'bot');
            });
        }

        function addMessage(message, sender) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}-message`;

            // Format newlines
            messageDiv.innerHTML = message.replace(/\\n/g, '<br>');

            chatbotMessages.appendChild(messageDiv);
            scrollToBottom();
        }

        function scrollToBottom() {
            chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
        }
    </script>
</body>
</html>"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    global model

    if model is None:
        return jsonify({'error': 'Model không được tải'}), 500

    try:
        # Lấy dữ liệu từ request
        data = request.json

        # Kiểm tra các trường bắt buộc
        required_fields = ['total_km', 'avg_km_per_trip', 'trips_per_day', 'vehicle_age_months']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Thiếu trường {field}'}), 400

        # Tạo DataFrame từ dữ liệu đầu vào
        input_data = pd.DataFrame([{
            "total_km": float(data['total_km']),
            "avg_km_per_trip": float(data['avg_km_per_trip']),
            "trips_per_day": int(data['trips_per_day']),
            "vehicle_age_months": int(data['vehicle_age_months'])
        }])

        # Kiểm tra loại mô hình và thực hiện dự đoán phù hợp
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        if isinstance(model, RandomForestClassifier):
            # Mô hình phân loại: dự đoán xem có cần bảo dưỡng hay không
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0] if hasattr(model, 'predict_proba') else None

            # Tính remaining_km dựa trên xác suất một cách chính xác hơn
            # Sử dụng xác suất để ước lượng khoảng cách đến kỳ bảo dưỡng
            if probabilities is not None:
                prob_maintenance = probabilities[1]  # xác suất cần bảo dưỡng

                # Chuyển đổi xác suất sang remaining_km theo logic kinh doanh
                # Nếu xác suất cao (>0.5), remaining_km thấp (<500)
                # Nếu xác suất thấp (<0.5), remaining_km cao (>500)
                if prob_maintenance > 0.5:
                    # Nếu xác suất cần bảo dưỡng cao, remaining_km sẽ thấp
                    estimated_remaining_km = 100 + (0.5 - prob_maintenance) * 800  # Giảm dần từ 500 xuống 100
                    estimated_remaining_km = max(100, estimated_remaining_km)  # Không dưới 100km
                else:
                    # Nếu xác suất cần bảo dưỡng thấp, remaining_km sẽ cao
                    estimated_remaining_km = 500 + (0.5 - prob_maintenance) * 7500  # Tăng dần từ 500 lên 8000
                    estimated_remaining_km = min(8000, estimated_remaining_km)  # Không quá 8000km
            else:
                # Nếu không có xác suất, sử dụng giá trị mặc định dựa trên dự đoán
                estimated_remaining_km = 300 if prediction == 1 else 6000  # 300km nếu cần bảo dưỡng, 6000km nếu không

            result = {
                'remaining_km': float(estimated_remaining_km),
                'needs_maintenance': bool(prediction),  # 1 nếu cần bảo dưỡng, 0 nếu không
                'status': 'Cần bảo dưỡng' if estimated_remaining_km <= 500 else 'Không cần bảo dưỡng'
            }

            if probabilities is not None:
                result['probability_no_maintenance'] = float(probabilities[0])
                result['probability_need_maintenance'] = float(probabilities[1])
        else:
            # Mô hình hồi quy: dự đoán số km còn lại
            remaining_km = model.predict(input_data)[0]
            result = {
                'remaining_km': float(remaining_km),
                'status': 'Cần bảo dưỡng' if remaining_km <= 500 else 'Không cần bảo dưỡng'
            }

        # Trả về kết quả
        return jsonify(result)

    except ValueError as ve:
        return jsonify({'error': f'Lỗi định dạng dữ liệu: {str(ve)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Lỗi xử lý: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')

        if not user_message:
            return jsonify({'response': 'Vui lòng nhập tin nhắn.'}), 400

        # Lấy phản hồi từ chatbot nâng cao
        bot_response = chatbot.chat(user_message)

        return jsonify({'response': bot_response})

    except Exception as e:
        return jsonify({'response': f'Có lỗi xảy ra: {str(e)}'}), 500

@app.route('/chat_maintenance', methods=['POST'])
def chat_maintenance():
    """
    API endpoint riêng cho chatbot bảo trì bảo dưỡng xe
    """
    try:
        data = request.json
        user_message = data.get('message', '')

        if not user_message:
            return jsonify({'response': 'Vui lòng nhập câu hỏi của bạn.'}), 400

        # Sử dụng chatbot hiện tại để xử lý câu hỏi về bảo trì bảo dưỡng
        # Tuy nhiên, chúng ta sẽ kiểm tra nếu câu hỏi liên quan đến bảo trì bảo dưỡng
        user_message_lower = user_message.lower()

        # Danh sách từ khóa liên quan đến bảo trì bảo dưỡng
        maintenance_keywords = [
            'bảo dưỡng', 'bảo duỡng', 'sửa chữa', 'hỏng', 'hỏng hóc', 'lốp', 'phanh', 'máy', 'động cơ',
            'gara', 'dầu nhớt', 'thay nhớt', 'thay dầu', 'lọc gió', 'bugi', 'ắc quy', 'xăng', 'nhiên liệu',
            'bình ắc quy', 'nhớt', 'dầu máy', 'dầu phanh', 'lọc nhớt', 'lọc xăng', 'lọc khí', 'nhông sên dĩa',
            'côn', 'ly hợp', 'hộp số', 'mát xe', 'nhiệt độ', 'nổ máy', 'đề máy', 'khởi động', 'tiếng ồn',
            'tiếng kêu', 'rung', 'giật', 'chập chờn', 'mất thắng', 'mất phanh', 'hỏng đèn', 'đèn không sáng',
            'hao xăng', 'hao nhiên liệu', 'tiêu hao nhiên liệu', 'bảo trì', 'kiểm tra', 'định kỳ', 'thời gian bảo dưỡng',
            'chi phí', 'giá sửa', 'giá thay', 'dịch vụ', 'trung tâm bảo dưỡng', 'trung tâm sửa chữa'
        ]

        # Kiểm tra xem câu hỏi có liên quan đến bảo trì bảo dưỡng không
        is_maintenance_related = any(keyword in user_message_lower for keyword in maintenance_keywords)

        if is_maintenance_related:
            # Nếu là câu hỏi về bảo trì bảo dưỡng, sử dụng chatbot với prompt chuyên môn
            bot_response = chatbot.chat(user_message)
        else:
            # Nếu không, vẫn xử lý nhưng hướng người dùng đến tính năng bảo trì bảo dưỡng
            bot_response = chatbot.chat(user_message)

        return jsonify({'response': bot_response})

    except Exception as e:
        return jsonify({'response': f'Có lỗi xảy ra khi xử lý yêu cầu: {str(e)}'}), 500

@app.route('/chatbot')
def chatbot_page():
    with open('chatbot_interface.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content

@app.route('/advanced_chatbot')
def advanced_chatbot_page():
    with open('advanced_chatbot_interface.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content

if __name__ == '__main__':
    app.run(debug=True, port=5000)