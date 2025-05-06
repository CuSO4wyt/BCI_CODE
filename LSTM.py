import numpy as np


# 1. 数据预处理（20行）
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 将心率数据规范化为统一长度
max_len = 6000  # 假设统一为60秒数据（100Hz采样）
X = []
for hr in heart_rates:
    if len(hr) >= max_len:
        X.append(hr[:max_len])
    else:
        X.append(np.pad(hr, (0, max_len - len(hr)), mode='constant'))
X = np.array(X)[..., np.newaxis]  # 添加通道维度 (n_samples, max_len, 1)

# 标签处理（每人一个标签）
y = np.array(labels)

# 2. 构建LSTM模型（10行）
model = Sequential([
    LSTM(64, input_shape=(max_len, 1)),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. 训练与评估（10行）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train, epochs=10, batch_size=8)
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")