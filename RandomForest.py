# 1. 数据预处理 + 特征工程（15行）
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 假设数据已加载为 heart_rates 和 labels
def extract_features(heart_rate, window_size=100):
    features = []
    for i in range(0, len(heart_rate) - window_size, window_size//2):  # 50%重叠窗口
        window = heart_rate[i:i+window_size]
        features.append([
            np.mean(window), np.std(window), 
            np.max(window)-np.min(window), 
            len(signal.find_peaks(window)[0])  # 峰值计数
        ])
    return np.array(features)

# 2. 训练模型（10行）
X = np.vstack([extract_features(hr) for hr in heart_rates])  # 所有被试的特征
y = np.hstack([np.full(len(f), label) for f, label in zip(X, labels)])  # 扩展标签

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 3. 评估（5行）
print("Accuracy:", model.score(X_test, y_test))