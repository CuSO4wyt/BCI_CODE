import pandas as pd
import numpy as np
from datetime import datetime

# --------- 读取 EEG 数据（txt 文件） ---------
eeg_file = 'EEG.txt'  # EEG 数据的 TXT 文件路径
eeg_df = pd.read_csv(eeg_file, sep=', ', header=None)  # 确保正确使用逗号分隔符

# 查看 EEG 数据格式
print("EEG 数据预览：")
print(eeg_df.head())

# --------- 时间戳解析 ---------
def convert_to_seconds(timestamp):
    time_obj = datetime.strptime(timestamp, "%H:%M:%S.%f")
    total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6
    return total_seconds

# 获取第一个时间戳
first_timestamp = convert_to_seconds(eeg_df[12][0])

# 将时间戳转化为相对时间戳（第一个时间戳为 0，后续为相对它的时间）
eeg_df['Relative_Time_Seconds'] = eeg_df[12].apply(lambda x: convert_to_seconds(x) - first_timestamp)

# 查看转换后的 EEG 数据
print("转换后的相对时间戳 EEG 数据：")
print(eeg_df.head())

# --------- EEG 信号特征提取 ---------
def extract_eeg_features_at_time(target_time, eeg_df):
    # 找到与目标时间最接近的时间戳
    closest_idx = (eeg_df['Relative_Time_Seconds'] - target_time).abs().argmin()
    
    # 提取最接近时间戳的 EEG 数据
    closest_eeg_data = eeg_df.iloc[closest_idx, 1:9].values  # 假设 EEG 数据在第二到第九列
    
    return closest_eeg_data

# --------- 与词语时间戳对齐 ---------
# 读取词语时间戳的 CSV 文件
word_timestamps_file = '6.csv'  # 词语时间戳 CSV 文件路径
word_df = pd.read_csv(word_timestamps_file)

# 查看词语时间戳数据
print("词语时间戳数据：")
print(word_df.head())

# 为每个词语找 EEG 时间点对应的特征
def align_eeg_with_word_time(word_df, eeg_df):
    aligned_data = []
    
    for _, word_row in word_df.iterrows():
        word_start_time = word_row['Start Time']
        word_end_time = word_row['End Time']
        
        # 计算词语中点时间
        word_mid_time = (word_start_time + word_end_time) / 2
        
        # 提取与中点时间最接近的 EEG 数据
        eeg_features = extract_eeg_features_at_time(word_mid_time, eeg_df)
        
        # 将词语的时间戳和 EEG 特征合并
        aligned_data.append({**word_row.to_dict(), **dict(zip([f"EEG_Channel_{i}" for i in range(1, 9)], eeg_features))})
    
    return pd.DataFrame(aligned_data)

# 将 EEG 特征与词语时间戳对齐
aligned_df = align_eeg_with_word_time(word_df, eeg_df)

# 查看合并后的数据
print("合并后的数据：")
print(aligned_df.head())

# 保存最终结果到 CSV 文件
output_file = 'multi.csv'
aligned_df.to_csv(output_file, index=False)

print(f"数据已保存到 {output_file}")
