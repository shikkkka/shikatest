from flask import Flask, request, render_template, send_file
import os
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import requests
import json

from typing import Optional

from 模型保存训练 import label_column

app = Flask(__name__)


# ================== IP溯源模块 ==================
class Street:
    def __init__(self, obj: dict):
        self.lng = obj.get("lng")
        self.lat = obj.get("lat")
        self.province = obj.get("province")
        self.city = obj.get("city")
        self.district = obj.get("district")
        self.street = obj.get("street")
        self.radius = obj.get("radius")
        self.zip_code = obj.get("zip_code")


class Location:
    def __init__(self, obj: dict):
        self.area_code = obj.get("area_code")
        self.city = obj.get("city")
        self.city_code = obj.get("city_code")
        self.continent = obj.get("continent")
        self.country = obj.get("country")
        self.country_code = obj.get("country_code")
        self.district = obj.get("district")
        self.elevation = obj.get("elevation")
        self.ip = obj.get("ip")
        self.isp = obj.get("isp")
        self.latitude = obj.get("latitude")
        self.longitude = obj.get("longitude")
        self.multi_street = [Street(s) for s in obj.get("multi_street", [])]
        self.province = obj.get("province")
        self.street = obj.get("street")
        self.time_zone = obj.get("time_zone")
        self.weather_station = obj.get("weather_station")
        self.zip_code = obj.get("zip_code")


IPDATA_CLOUD_API_KEY = "76f551e71c2b11f0b3fd00163e167ffb"


def query_ipdatacloud(ip: str) -> Optional[Location]:
    """IP地理位置查询"""
    base_url = "https://api.ipdatacloud.com/v2/query"
    try:
        response = requests.get(base_url, params={
            "ip": ip,
            "key": IPDATA_CLOUD_API_KEY,
            "lang": "CN"
        }, timeout=5)
        data = response.json()
        if data.get("code") == 200:
            return Location(data["data"]["location"])
        return None
    except Exception as e:
        print(f"IP查询失败: {str(e)}")
        return None


# ================== 流量预测模块 ==================
# 模型配置
selected_features = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Bwd Packet Length Max', 'Bwd Packet Length Min',
    'Flow Bytes/s', 'Flow Packets/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Fwd IAT Mean', 'Bwd IAT Mean',
    'FIN Flag Count', 'SYN Flag Count', 'ACK Flag Count',
    'Protocol',
    'Packet Length Mean', 'Packet Length Std',
    'Active Mean', 'Idle Mean'  # 确保包含所有特征列
]

def load_models():
    """加载预处理工具和模型"""
    preprocessor = joblib.load('feature_preprocessor.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')

    # 获取特征列数量
    input_dim = len(selected_features)

    # 重建模型结构（确保输入维度与特征列数量一致）
    model = Sequential([
        LSTM(128, activation='relu', input_shape=(1, input_dim)),  # 动态设置输入维度
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 加载权重
    model_weights_path = 'optimized_lstm_model_weights.h5'
    if os.path.exists(model_weights_path):
        print("加载已有模型权重...")
        model.load_weights(model_weights_path)
    else:
        raise FileNotFoundError("未找到模型权重文件，请确保模型已训练并保存权重。")

    return preprocessor, scaler, label_encoder, model

preprocessor, scaler, label_encoder, model = load_models()


def predict_traffic(new_data: pd.DataFrame) -> pd.DataFrame:
    """执行流量预测和IP溯源"""
    # 分离IP和特征
    src_ips = new_data['Src IP'].copy()
    model_data = new_data.drop(columns=['Src IP', label_column], errors='ignore')

    # 数据预处理
    processed = preprocessor.transform(model_data)
    scaled = scaler.transform(processed)
    reshaped = scaled.reshape((scaled.shape[0], 1, scaled.shape[1]))

    # 预测
    preds = model.predict(reshaped, verbose=0)
    labels = label_encoder.inverse_transform(np.argmax(preds, axis=1))
    confidences = np.max(preds, axis=1)

    # 构建结果
    result = pd.DataFrame({
        'Src IP': src_ips,
        'Predicted_Label': labels,
        'Confidence': confidences,
        'Is_Malicious': labels != 'Benign'  # 根据实际标签调整
    })

    # IP溯源
    geo_data = []
    for ip in src_ips.unique():
        location = query_ipdatacloud(ip)
        if location:
            geo_data.append({
                'Src IP': ip,
                'Country': location.country,
                'City': location.city,
                'ISP': location.isp,
                'Coordinates': f"{location.latitude},{location.longitude}"
            })

    if geo_data:
        geo_df = pd.DataFrame(geo_data)
        result = pd.merge(result, geo_df, on='Src IP', how='left')
    else:
        result[['Country', 'City', 'ISP', 'Coordinates']] = np.nan

    return result


# ================== Flask路由 ==================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "请选择文件", 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return "仅支持CSV文件", 400

    # 保存文件
    os.makedirs('uploads', exist_ok=True)
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    try:
        # 读取数据
        data = pd.read_csv(file_path, encoding='gbk', usecols=selected_features + [label_column])
    except Exception as e:
        return f"文件读取失败: {str(e)}", 400

    # 执行预测
    try:
        result = predict_traffic(data)
    except Exception as e:
        return f"预测失败: {str(e)}", 500

    # 保存结果
    os.makedirs('outputs', exist_ok=True)
    output_path = os.path.join('outputs', 'result.csv')
    result.to_csv(output_path, index=False)

    # 准备展示数据
    display_df = result.head(10).fillna('N/A')
    html_table = display_df.to_html(
        index=False,
        formatters={
            'Coordinates': lambda
                x: f'<a href="https://maps.google.com/?q={x}" target="_blank">{x}</a>' if x != 'N/A' else '',
            'Is_Malicious': lambda x: '⚠️' if x else '✅'
        },
        escape=False,
        classes='result-table'
    )

    return render_template('result.html',
                           table=html_table,
                           total=len(result))


@app.route('/download')
def download():
    return send_file('outputs/result.csv', as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, port=5000)