import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 读取数据
df = pd.read_csv("multi.csv")

# 确保所有输入特征数组的样本数一致
df = df.dropna()  # 删除包含缺失值的样本

# 获取输入特征（例如MFCC、面部特征、文本特征等）
eeg_columns = [f"EEG_Channel_{i}" for i in range(1, 8)]
mfcc_columns = [f"MFCC_{i}" for i in range(1, 14)]
face_columns = ["Left Eye X", "Left Eye Y", "Mouth X", "Mouth Y", "Eyebrow X", "Eyebrow Y", 
                "Chin X", "Chin Y", "Smile Intensity", "Frown Intensity", "Eye Closure", 
                "AU6", "AU12", "AU1_2"]
text_columns = [str(i) for i in range(0, 768)]  # BERT特征的列名是数字
hr_columns = ['HR_Avg']  # 心率
spo2_columns = ['SpO2_Avg']  # 血氧
labels = ['Label']  # 标签

# 提取数据
X_eeg = df[eeg_columns].values
X_mfcc = df[mfcc_columns].values
X_face = df[face_columns].values
X_text = df[text_columns].values
X_hr = df[hr_columns].values
X_spo2 = df[spo2_columns].values
y = df[labels].values

# 归一化/标准化处理
scaler = StandardScaler()

X_eeg = scaler.fit_transform(X_eeg)
X_mfcc = scaler.fit_transform(X_mfcc)
X_face = scaler.fit_transform(X_face)
X_text = scaler.fit_transform(X_text)
X_hr = scaler.fit_transform(X_hr)
X_spo2 = scaler.fit_transform(X_spo2)

# 合并所有特征数据
X_combined = np.concatenate([X_eeg, X_mfcc, X_face, X_text, X_hr, X_spo2], axis=1)

# 数据集分割
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# 将X_train拆分为相应的输入数据
X_train_eeg = X_train[:, :X_eeg.shape[1]]  # EEG特征
X_train_mfcc = X_train[:, X_eeg.shape[1]:X_eeg.shape[1] + X_mfcc.shape[1]]  # MFCC特征
X_train_face = X_train[:, X_eeg.shape[1] + X_mfcc.shape[1]:X_eeg.shape[1] + X_mfcc.shape[1] + X_face.shape[1]]  # 面部特征
X_train_text = X_train[:, X_eeg.shape[1] + X_mfcc.shape[1] + X_face.shape[1]:X_eeg.shape[1] + X_mfcc.shape[1] + X_face.shape[1] + X_text.shape[1]]  # 文本特征
X_train_hr = X_train[:, X_eeg.shape[1] + X_mfcc.shape[1] + X_face.shape[1] + X_text.shape[1]:X_eeg.shape[1] + X_mfcc.shape[1] + X_face.shape[1] + X_text.shape[1] + X_hr.shape[1]]  # 心率特征
X_train_spo2 = X_train[:, X_eeg.shape[1] + X_mfcc.shape[1] + X_face.shape[1] + X_text.shape[1] + X_hr.shape[1]:]  # 血氧特征


X_test_eeg = X_test[:, :X_eeg.shape[1]]  # EEG特征
X_test_mfcc = X_test[:, X_eeg.shape[1]:X_eeg.shape[1] + X_mfcc.shape[1]]  # MFCC特征
X_test_face = X_test[:, X_eeg.shape[1] + X_mfcc.shape[1]:X_eeg.shape[1] + X_mfcc.shape[1] + X_face.shape[1]]  # 面部特征
X_test_text = X_test[:, X_eeg.shape[1] + X_mfcc.shape[1] + X_face.shape[1]:X_eeg.shape[1] + X_mfcc.shape[1] + X_face.shape[1] + X_text.shape[1]]  # 文本特征
X_test_hr = X_test[:, X_eeg.shape[1] + X_mfcc.shape[1] + X_face.shape[1] + X_text.shape[1]:X_eeg.shape[1] + X_mfcc.shape[1] + X_face.shape[1] + X_text.shape[1] + X_hr.shape[1]]  # 心率特征
X_test_spo2 = X_test[:, X_eeg.shape[1] + X_mfcc.shape[1] + X_face.shape[1] + X_text.shape[1] + X_hr.shape[1]:]  # 血氧特征

from tensorflow.keras import layers, models
from tensorflow.keras.models import Model

# EEG模型 - 使用LSTM处理时间序列数据
def build_eeg_model(input_shape=(100, 1)):
    model = models.Sequential([
        layers.LSTM(64, input_shape=input_shape, return_sequences=False),
        layers.Dense(128, activation='relu'),
    ])
    return model

# 面部表情模型 - 使用Dense层处理面部特征（修改为Dense层）
def build_face_model(input_shape=(14,)):  # 修改输入形状为一维数据
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dense(128, activation='relu'),
    ])
    return model

# 语音处理模型 - 使用Dense层处理MFCC特征
def build_audio_model(input_shape=(13,)):
    input_layer = layers.Input(shape=input_shape)
    x = layers.Dense(64, activation='relu')(input_layer)
    x = layers.Dense(128, activation='relu')(x)
    return models.Model(inputs=input_layer, outputs=x)

# 文本处理模型 - 使用BERT处理文本特征
def build_text_model(input_shape=(768,)):
    input_layer = layers.Input(shape=input_shape)
    x = layers.Dense(128, activation='relu')(input_layer)
    return models.Model(inputs=input_layer, outputs=x)

# 融合模型 - 将各模态的输出融合并进行分类
def build_multimodal_model(eeg_input_shape, face_input_shape, audio_input_shape, text_input_shape):
    eeg_model = build_eeg_model(eeg_input_shape)
    face_model = build_face_model(face_input_shape)
    audio_model = build_audio_model(audio_input_shape)
    text_model = build_text_model(text_input_shape)

    # 融合各模态的输出
    combined = layers.concatenate([eeg_model.output, face_model.output, audio_model.output, text_model.output])

    # 全连接层
    x = layers.Dense(256, activation='relu')(combined)
    x = layers.Dense(128, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    # 构建最终模型
    model = models.Model(inputs=[eeg_model.input, face_model.input, audio_model.input, text_model.input], outputs=output)

    return model



# 输入形状
eeg_input_shape = (X_eeg.shape[1], 1)
face_input_shape = (X_face.shape[1],)
audio_input_shape = (X_mfcc.shape[1],)
text_input_shape = (X_text.shape[1],)


# 构建模型
model = build_multimodal_model(eeg_input_shape, face_input_shape, audio_input_shape, text_input_shape)

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
# 训练模型时传入合并后的数据
model.fit(
    [X_train_eeg, X_train_face, X_train_mfcc, X_train_text],  # 传入各个特征的拆分数据
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=([X_test_eeg, X_test_face, X_test_mfcc, X_test_text], y_test)  # 传入测试数据
)

# 评估模型
# 评估模型时，传入拆分后的测试数据
loss, accuracy = model.evaluate(
    [X_test_eeg, X_test_face, X_test_mfcc, X_test_text],  # 传入拆分后的测试数据
    y_test
)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")
