import os
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def load_trained_models():
    """加载训练好的预处理工具和模型"""
    # 加载预处理管道
    preprocessor = joblib.load('feature_preprocessor.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')

    # 直接从预处理器中获取输入维度
    input_dim = preprocessor.transformers_[0][1].get_feature_names_out().shape[0] + preprocessor.transformers_[1][1].n_features_in_
    timesteps = 1

    # 重建模型结构（必须与训练代码完全一致）
    model = Sequential([
        LSTM(128, activation='relu', input_shape=(timesteps, input_dim), return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),  # 修改为 64
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 加载权重
    model_weights_path = 'optimized_lstm_model_weights.h5'
    if os.path.exists(model_weights_path):
        print("加载已有模型权重继续训练...")
        model.load_weights(model_weights_path)
    else:
        print("未找到模型权重文件，将从头开始训练模型...")

    return preprocessor, scaler, label_encoder, model


# 确保与训练时完全一致的特征列（不包含Label）
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


def predict_traffic(input_csv, output_csv):
    """执行流量预测并保存结果"""
    # 加载新数据（保持编码一致）
    try:
        new_data = pd.read_csv(input_csv, usecols=selected_features, encoding='gbk')
    except ValueError as e:
        missing_cols = set(selected_features) - set(pd.read_csv(input_csv, nrows=0, encoding='gbk').columns)
        raise ValueError(f"输入文件缺少必要列: {missing_cols}") from e

    # 处理无效值
    new_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    if new_data.isnull().sum().sum() > 0:
        print(f"警告: 发现 {new_data.isnull().sum().sum()} 个空值，已自动删除")
        new_data.dropna(inplace=True)

    # 加载模型和预处理工具
    preprocessor, scaler, label_encoder, model = load_trained_models()

    # 应用预处理
    try:
        processed_features = preprocessor.transform(new_data)
    except ValueError as e:
        raise RuntimeError("预处理失败，请检查Protocol列是否包含训练时未见的值") from e

    scaled_features = scaler.transform(processed_features)

    # 调整输入形状
    timesteps = 1
    X_new = scaled_features.reshape((scaled_features.shape[0], timesteps, scaled_features.shape[1]))

    # 预测
    y_pred = model.predict(X_new, verbose=0)
    y_pred_labels = label_encoder.inverse_transform(np.argmax(y_pred, axis=1))
    confidences = np.max(y_pred, axis=1)

    # 添加预测结果
    new_data['Predicted_Label'] = y_pred_labels
    new_data['Confidence'] = confidences

    # 保存结果
    new_data.to_csv(output_csv, index=False)
    print(f"预测完成，结果保存至: {output_csv}")
    return new_data


# 示例用法
if __name__ == "__main__":
    input_file = "F:/test/S7.pcap_Flow.csv"
    output_file = "F:/test/shuchu.csv"
    result = predict_traffic(input_file, output_file)
    print("\n预测结果示例:")
    print(result[['Flow Duration', 'Protocol', 'Predicted_Label', 'Confidence']].head())
