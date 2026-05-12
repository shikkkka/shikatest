import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint

# 定义要保留的特征列
selected_features = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Bwd Packet Length Max', 'Bwd Packet Length Min',
    'Flow Bytes/s', 'Flow Packets/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Fwd IAT Mean', 'Bwd IAT Mean',
    'FIN Flag Count', 'SYN Flag Count', 'ACK Flag Count',
    'Protocol',  # 需One-Hot编码
    'Packet Length Mean', 'Packet Length Std',
    'Active Mean', 'Idle Mean',
    'Label'  # 标签列必须保留
]

# 加载数据并筛选列
raw_data_filename = "F:/毕设文件/终版/zhongdata.csv"
print("Loading raw data...")
raw_data = pd.read_csv(raw_data_filename, usecols=selected_features, low_memory=False)

# 随机抽样
raw_data = raw_data.sample(frac=0.8)

# 处理无穷大和NaN值
raw_data.replace([np.inf, -np.inf], np.nan, inplace=True)
raw_data.dropna(inplace=True)

# 检查标签分布
print("Label distribution:")
print(raw_data['Label'].value_counts())

# 分离特征和标签
labels = raw_data['Label']
features = raw_data.drop(columns=['Label'])

# 处理Protocol列（One-Hot编码）
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), ['Protocol']),  # 编码Protocol
        ('num', 'passthrough', features.columns.difference(['Protocol']))  # 其他数值列保留
    ],
    remainder='drop'
)

# 应用预处理
features_processed = preprocessor.fit_transform(features)

# 标签编码
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# 标准化数值特征（注意：One-Hot编码后的Protocol列不需要标准化）
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

# SMOTE过采样（注意：需先展平数据）
smote = SMOTE(random_state=42)
X_train_flat = X_train_reshaped.reshape(X_train_reshaped.shape[0], -1)
X_resampled, y_resampled = smote.fit_resample(X_train_flat, y_train_onehot)
X_resampled_reshaped = X_resampled.reshape((X_resampled.shape[0], timesteps, X_resampled.shape[1]))

# 构建优化后的LSTM模型
input_dim = X_resampled_reshaped.shape[2]
model = Sequential([
    LSTM(64, activation='relu', input_shape=(timesteps, input_dim), return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(
    X_resampled_reshaped, y_resampled,
    epochs=2,
    batch_size=128,
    validation_data=(X_test_reshaped, y_test_onehot),
    verbose=1
)

# 评估模型
y_pred = model.predict(X_test_reshaped)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test_onehot, axis=1)
accuracy = accuracy_score(y_true_classes, y_pred_classes)
print(f"模型准确率: {accuracy:.4f}")

# 保存模型和预处理对象
model.save('optimized_lstm_model.h5')
joblib.dump(preprocessor, 'feature_preprocessor.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')