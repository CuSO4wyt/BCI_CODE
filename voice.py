import pandas as pd
import librosa
import numpy as np

# 提取音频文件的 MFCC 特征
def extract_mfcc(audio_file, start_time, end_time, sr=16000):
    # 加载音频文件，返回音频数据和采样率
    audio, sr = librosa.load(audio_file, sr=sr)  # 设置采样率为 16kHz

    
    # 计算对应的样本索引
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    
    # 截取音频数据
    audio_segment = audio[start_sample:end_sample]
    
    # 提取 MFCC 特征
    mfcc = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=13)  # 提取 13 个 MFCC 特征
    mfcc_mean = np.mean(mfcc, axis=1)  # 对每个特征维度取均值，减少数据维度

    return mfcc_mean

# 读取已有的 CSV 文件
input_file = '1.csv'  # 输入的 CSV 文件路径
df = pd.read_csv(input_file)

# 假设音频文件是 'example.wav'
audio_file = 'converted.wav'

# 为每一行添加 MFCC 特征列
mfcc_features = []

for _, row in df.iterrows():
    start_time = row['Start Time']
    end_time = row['End Time']
    
    # 提取当前词对应的 MFCC 特征
    mfcc = extract_mfcc(audio_file, start_time, end_time)
    mfcc_features.append(mfcc)

# 将 MFCC 特征列添加到 DataFrame 中
mfcc_features_df = pd.DataFrame(mfcc_features, columns=[f"MFCC_{i+1}" for i in range(13)])  # 13 个 MFCC 特征
df = pd.concat([df, mfcc_features_df], axis=1)

# 保存更新后的 CSV 文件
output_file = '2.csv'  # 输出文件路径
df.to_csv(output_file, index=False)

print(f"数据已保存到 {output_file}")
