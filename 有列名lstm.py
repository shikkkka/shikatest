import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, zero_one_loss, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# 加载数据
raw_data_filename = "F:\毕设文件\数据集\机器学习data\有列名.csv"
print("Loading raw data...")
raw_data = pd.read_csv(raw_data_filename, low_memory=False)

# 随机抽取比例，当数据集比较大的时候，可以采用这个，可选项
raw_data = raw_data.sample(frac=0.1)

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
features = preprocessing.scale(features)

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

# 构建LSTM模型
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(timesteps, X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
print("Training model...")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 预测
print("Predicting...")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# 评估准确率
accuracy = accuracy_score(y_true_classes, y_pred_classes)
print(f"模型准确率: {accuracy:.4f}")

# 计算混淆矩阵和0-1损失
results = confusion_matrix(y_true_classes, y_pred_classes)
error = zero_one_loss(y_true_classes, y_pred_classes)

# 根据混淆矩阵求预测精度
list_diag = np.diag(results)
list_raw_sum = np.sum(results, axis=1)
print("Predict accuracy of the LSTM: ", np.mean(list_diag) / np.mean(list_raw_sum))
