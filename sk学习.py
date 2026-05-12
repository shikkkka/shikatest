import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# 1. 加载数据（替换为你的实际数据）
data = pd.read_csv('F:\毕设文件\数据集\机器学习data\Monday-WorkingHours.pcap_ISCX.csv')

# 2. 数值特征
numeric_features = ['Source Port', 'Destination Port', 'Packet Length']
X_numeric = data[numeric_features]

# 3. 类别特征编码
protocol_encoder = LabelEncoder()
X_protocol = protocol_encoder.fit_transform(data['Protocol'])

flags_encoder = OneHotEncoder(sparse=False)
X_flags = flags_encoder.fit_transform(data[['Flags']])

# 4. 时间戳处理
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
X_time = data['Timestamp'].astype('int64') // 10**9
hour = data['Timestamp'].dt.hour
data['Hour_sin'] = np.sin(2 * np.pi * hour / 24)
data['Hour_cos'] = np.cos(2 * np.pi * hour / 24)

# 5. 流量统计特征
flow_stats = data.groupby('Source IP').agg({
    'Packet Length': ['sum', 'count', 'mean', 'std'],
    'Destination Port': 'nunique'
}).reset_index()
flow_stats.columns = [
    'Source IP', 'Total Bytes', 'Packet Count',
    'Avg Packet Size', 'Packet Size Std', 'Unique Dest Ports'
]
data = data.merge(flow_stats, on='Source IP', how='left')

# 6. 合并特征
X = pd.concat([
    X_numeric,
    pd.Series(X_protocol, name='Protocol'),
    pd.DataFrame(X_flags, columns=flags_encoder.get_feature_names_out(['Flags'])),
    data[['Total Bytes', 'Packet Count', 'Hour_sin', 'Hour_cos']]
], axis=1)

# 7. 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8. 保存特征矩阵
feature_df = pd.DataFrame(X_scaled, columns=X.columns)
feature_df['Label'] = data['Label']  # 保留标签列
feature_df.to_csv('extracted_features.csv', index=False)