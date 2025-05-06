import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout

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

# 如果 Start Time 和 End Time 列需要作为时间特征，可以这样处理
start_time = df['Start Time'].values
end_time = df['End Time'].values
# 你可以对 start_time 和 end_time 做一些处理，例如计算时间间隔等，假设我们将其添加为额外的特征：
time_diff = end_time - start_time  # 时间差，假设这是你想要的特征
X_time = time_diff.reshape(-1, 1)  # 将时间差reshape为一列

# 归一化/标准化处理
scaler = StandardScaler()

X_eeg = scaler.fit_transform(X_eeg)
X_mfcc = scaler.fit_transform(X_mfcc)
X_face = scaler.fit_transform(X_face)
X_text = scaler.fit_transform(X_text)
X_hr = scaler.fit_transform(X_hr)
X_spo2 = scaler.fit_transform(X_spo2)
X_time = scaler.fit_transform(X_time)  # 对时间差特征做标准化

# 合并所有特征数据
X_combined = np.concatenate([X_eeg, X_mfcc, X_face, X_text, X_hr, X_spo2, X_time], axis=1)

# 数据集分割
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# 将X_train拆分为相应的输入数据
X_train_eeg = X_train[:, :X_eeg.shape[1]]  # EEG特征
X_train_mfcc = X_train[:, X_eeg.shape[1]:X_eeg.shape[1] + X_mfcc.shape[1]]  # MFCC特征
X_train_face = X_train[:, X_eeg.shape[1] + X_mfcc.shape[1]:X_eeg.shape[1] + X_mfcc.shape[1] + X_face.shape[1]]  # 面部特征
X_train_text = X_train[:, X_eeg.shape[1] + X_mfcc.shape[1] + X_face.shape[1]:X_eeg.shape[1] + X_mfcc.shape[1] + X_face.shape[1] + X_text.shape[1]]  # 文本特征
X_train_hr = X_train[:, X_eeg.shape[1] + X_mfcc.shape[1] + X_face.shape[1] + X_text.shape[1]:X_eeg.shape[1] + X_mfcc.shape[1] + X_face.shape[1] + X_text.shape[1] + X_hr.shape[1]]  # 心率特征
X_train_spo2 = X_train[:, X_eeg.shape[1] + X_mfcc.shape[1] + X_face.shape[1] + X_text.shape[1] + X_hr.shape[1]:X_eeg.shape[1] + X_mfcc.shape[1] + X_face.shape[1] + X_text.shape[1] + X_hr.shape[1] + X_spo2.shape[1]]  # 血氧特征
X_train_time = X_train[:, -1:]  # 时间差特征


X_test_eeg = X_test[:, :X_eeg.shape[1]]  # EEG特征
X_test_mfcc = X_test[:, X_eeg.shape[1]:X_eeg.shape[1] + X_mfcc.shape[1]]  # MFCC特征
X_test_face = X_test[:, X_eeg.shape[1] + X_mfcc.shape[1]:X_eeg.shape[1] + X_mfcc.shape[1] + X_face.shape[1]]  # 面部特征
X_test_text = X_test[:, X_eeg.shape[1] + X_mfcc.shape[1] + X_face.shape[1]:X_eeg.shape[1] + X_mfcc.shape[1] + X_face.shape[1] + X_text.shape[1]]  # 文本特征
X_test_hr = X_test[:, X_eeg.shape[1] + X_mfcc.shape[1] + X_face.shape[1] + X_text.shape[1]:X_eeg.shape[1] + X_mfcc.shape[1] + X_face.shape[1] + X_text.shape[1] + X_hr.shape[1]]  # 心率特征
X_test_spo2 = X_test[:, X_eeg.shape[1] + X_mfcc.shape[1] + X_face.shape[1] + X_text.shape[1] + X_hr.shape[1]:X_eeg.shape[1] + X_mfcc.shape[1] + X_face.shape[1] + X_text.shape[1] + X_hr.shape[1] + X_spo2.shape[1]]  # 血氧特征
X_test_time = X_test[:, -1:]  # 时间差特征


# Transformer层
def transformer_layer(inputs, head_size, num_heads, ff_dim, dropout_rate=0.1):
    # 确保输入是三维的： (batch_size, sequence_length, feature_dimension)
    if len(inputs.shape) == 2:
        inputs = layers.Reshape((-1, inputs.shape[-1]))(inputs)  # 将二维输入转为三维

    attention = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    attention = Dropout(dropout_rate)(attention)
    attention = LayerNormalization(epsilon=1e-6)(attention + inputs)  # 残差连接
    
    # 前馈神经网络（Feed Forward Network）
    ff = layers.Dense(ff_dim, activation='relu')(attention)
    ff = Dropout(dropout_rate)(ff)
    ff = layers.Dense(inputs.shape[-1])(ff)
    
    # 再次应用Layer Normalization
    return LayerNormalization(epsilon=1e-6)(ff + attention)  # 残差连接


# EEG模型 - 使用Transformer处理序列数据
def build_eeg_model(input_shape=(100, 1), head_size=64, num_heads=4, ff_dim=128):
    input_layer = layers.Input(shape=input_shape)
    x = transformer_layer(input_layer, head_size, num_heads, ff_dim)
    x = layers.GlobalAveragePooling1D()(x)  # 使用全局池化
    x = layers.Dense(128, activation='relu')(x)
    return models.Model(inputs=input_layer, outputs=x)

# 语音处理模型（MFCC） - 使用Transformer处理
def build_audio_model(input_shape=(13,), head_size=64, num_heads=4, ff_dim=128):
    input_layer = layers.Input(shape=input_shape)
    
    # 添加时间维度（假设时间步为1）
    x = layers.Reshape((-1, input_shape[-1]))(input_layer)  # 转换为三维输入，(batch_size, 1, feature_dimension)
    
    x = transformer_layer(x, head_size, num_heads, ff_dim)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    return models.Model(inputs=input_layer, outputs=x)

# 为文本特征创建一个模型
def build_text_model(input_shape=(768,), head_size=64, num_heads=4, ff_dim=128):
    input_layer = layers.Input(shape=input_shape)
    
    # 添加时间维度
    x = layers.Reshape((-1, input_shape[-1]))(input_layer)
    
    x = transformer_layer(x, head_size, num_heads, ff_dim)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    return models.Model(inputs=input_layer, outputs=x)

# 面部表情模型 - 使用Dense层处理面部特征（修改为Dense层）
def build_face_model(input_shape=(14,)):  # 修改输入形状为一维数据
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dense(128, activation='relu'),
    ])
    return model

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
model.fit(
    [X_train_eeg, X_train_face, X_train_mfcc, X_train_text],  # 传入各个特征的拆分数据
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=([X_test_eeg, X_test_face, X_test_mfcc, X_test_text], y_test)  # 传入测试数据
)

# 评估模型
loss, accuracy = model.evaluate(
    [X_test_eeg, X_test_face, X_test_mfcc, X_test_text],  # 传入拆分后的测试数据
    y_test
)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")
