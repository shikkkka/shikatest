import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, zero_one_loss, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE

# 加载数据
raw_data_filename = "F:\毕设文件\数据集\机器学习data\有列名.csv"
print("Loading raw data...")
raw_data = pd.read_csv(raw_data_filename, low_memory=False)

# 随机抽取比例，当数据集比较大的时候，可以采用这个，可选项
raw_data = raw_data.sample(frac=0.2)

# 检查数据中的无穷大值和NaN值
print("Checking for infinity and NaN values in raw data:")
print(raw_data.isin([np.inf, -np.inf]).sum().sum())  # 统计无穷大值的数量
print(raw_data.isnull().sum().sum())  # 统计NaN值的数量

# 处理无穷大和NaN值
raw_data.replace([np.inf, -np.inf], np.nan, inplace=True)
raw_data.dropna(inplace=True)

# 查看标签数据情况
print("print data labels:")
print(raw_data[' Label'].value_counts())

# 将非数值型的数据转换为数值型数据
label_encoder = LabelEncoder()
raw_data[' Label'] = label_encoder.fit_transform(raw_data[' Label'])

# 分离出特征和标签
features = raw_data.drop(columns=[' Label'])
labels = raw_data[' Label']

# 特征数据标准化
scaler = preprocessing.StandardScaler()
features = scaler.fit_transform(features)

# 将多维的标签转为一维的数组
labels = labels.values

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.8, test_size=0.2, stratify=labels)

# 重塑数据以适应LSTM输入 (samples, timesteps, features)
timesteps = 1  # 这里假设每个样本的时间步长为1，可以根据实际情况调整
X_train = X_train.reshape((X_train.shape[0], timesteps, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], timesteps, X_test.shape[1]))

# 将标签转换为one-hot编码
num_classes = len(np.unique(labels))
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# 过采样
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train.reshape(X_train.shape[0], -1), y_train)
X_train_resampled = X_train_resampled.reshape((X_train_resampled.shape[0], timesteps, X_train_resampled.shape[1]))

# 构建LSTM模型
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(timesteps, X_train_resampled.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
print("Training model...")
model.fit(X_train_resampled, y_train_resampled, epochs=20, batch_size=64, validation_data=(X_test, y_test))

# 保存模型
model.save('lstm_model.h5')

# 加载模型
model = load_model('lstm_model.h5')

# 预处理输入数据
def preprocess_input_data(input_data, scaler):
    # 处理无穷大和NaN值
    input_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    input_data.dropna(inplace=True)

    # 分离特征和标签（如果有标签）
    features = input_data.drop(columns=[' Label'])
    labels = input_data[' Label'] if ' Label' in input_data.columns else None

    # 特征数据标准化
    features = scaler.transform(features)

    # 重塑数据以适应LSTM输入 (samples, timesteps, features)
    timesteps = 1
    features = features.reshape((features.shape[0], timesteps, features.shape[1]))

    # 将标签转换为one-hot编码（如果有标签）
    if labels is not None:
        labels = label_encoder.transform(labels)
        labels = to_categorical(labels, num_classes)

    return features, labels

# 进行预测
def predict_with_model(model, input_data, scaler):
    features, _ = preprocess_input_data(input_data, scaler)
    y_pred = model.predict(features)
    y_pred_classes = np.argmax(y_pred, axis=1)
    return y_pred_classes

# 计算准确率
def calculate_accuracy(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    print(f"模型准确率: {accuracy:.4f}")
    return accuracy

# 示例输入数据
new_data_filename = "F:\毕设文件\数据集\机器学习data\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
print("Loading new data...")
new_data = pd.read_csv(new_data_filename, low_memory=False)

# 预处理新数据
new_features, new_labels = preprocess_input_data(new_data, scaler)

# 进行预测
new_y_pred_classes = predict_with_model(model, new_data, scaler)

# 如果有真实标签，计算准确率
if new_labels is not None:
    new_y_true_classes = np.argmax(new_labels, axis=1)
    calculate_accuracy(new_y_true_classes, new_y_pred_classes)
else:
    print("没有真实标签，无法计算准确率。")
