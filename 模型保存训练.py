import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ==== 关键修改1：明确分离特征列和标签列 ====
# 特征列（必须与预测代码完全一致，不含Label）
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
    'Active Mean', 'Idle Mean'  # 不含Label
]
label_column = 'Label'  # 单独定义标签列

# ==== 关键修改2：加载数据时明确指定特征列 ====
raw_data_filename = "F:/毕设文件/终版/zhongdata.csv"
print("Loading raw data...")
raw_data = pd.read_csv(
    raw_data_filename,
    usecols=selected_features + [label_column],  # 明确加载特征列+标签列
    low_memory=False
)

# 随机抽样
raw_data = raw_data.sample(frac=1)

# 处理无穷大和NaN值
raw_data.replace([np.inf, -np.inf], np.nan, inplace=True)
raw_data.dropna(inplace=True)

# 检查标签分布
print("Label distribution:")
print(raw_data[label_column].value_counts())

# 分离特征和标签
labels = raw_data[label_column]
features = raw_data[selected_features]  # 确保只保留特征列

# ==== 关键修改3：增强预处理鲁棒性 ====
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), ['Protocol']),  # 处理未知协议
    ],
    remainder='passthrough'  # 其他列直接保留
)

# 应用预处理
features_processed = preprocessor.fit_transform(features)

# 标签编码
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# 标准化数值特征
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_processed)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, labels_encoded,
    test_size=0.1, stratify=labels_encoded, random_state=42
)

# 调整LSTM输入形状
timesteps = 1
X_train_reshaped = X_train.reshape((X_train.shape[0], timesteps, X_train.shape[1]))
X_test_reshaped = X_test.reshape((X_test.shape[0], timesteps, X_test.shape[1]))

# One-Hot编码标签
num_classes = len(np.unique(labels_encoded))
y_train_onehot = to_categorical(y_train, num_classes)
y_test_onehot = to_categorical(y_test, num_classes)

# SMOTE过采样
smote = SMOTE(random_state=42)
X_train_flat = X_train_reshaped.reshape(X_train_reshaped.shape[0], -1)
X_resampled, y_resampled = smote.fit_resample(X_train_flat, y_train_onehot)
X_resampled_reshaped = X_resampled.reshape((X_resampled.shape[0], timesteps, X_resampled.shape[1]))

# 模型配置
model_weights_path = 'optimized_lstm_model_weights.h5'
model = Sequential([
    LSTM(128, activation='relu', input_shape=(timesteps, X_resampled_reshaped.shape[2])),  # 增加神经元
    Dropout(0.4),
    Dense(64, activation='relu', kernel_regularizer='l2'),  # 添加正则化
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 断点续训（自动加载已有权重）
if os.path.exists(model_weights_path):
    print("加载已有模型权重继续训练...")
    model.load_weights(model_weights_path)

# 训练模型（增加训练轮次）
history = model.fit(
    X_resampled_reshaped, y_resampled,
    epochs=2,  # 增加至20轮
    batch_size=256,  # 增大批次
    validation_data=(X_test_reshaped, y_test_onehot),
    callbacks=[
        EarlyStopping(monitor='val_accuracy', patience=5),  # 早停
        ModelCheckpoint(model_weights_path, save_best_only=True)  # 自动保存最佳模型
    ],
    verbose=1
)

# 评估模型
y_pred = model.predict(X_test_reshaped)
y_pred_classes = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(np.argmax(y_test_onehot, axis=1), y_pred_classes)
print(f"验证集准确率: {accuracy:.4f}")
model.save('full_model.h5')  # 新增这行

# 保存预处理管道
joblib.dump(preprocessor, 'feature_preprocessor.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
print("模型和预处理管道保存完毕")
