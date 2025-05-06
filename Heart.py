import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

# --------- 参数配置 ---------
FS = 200                  # 采样率 (200Hz = 1/0.005s)
LOW_CUT = 0.5             # 心率滤波低频 (Hz)
HIGH_CUT = 4.0            # 心率滤波高频 (Hz)
WINDOW_SIZE = 5           # 分析窗口大小 (秒)
STEP_SIZE = 1             # 滑动步长 (秒)
SPO2_R_MIN = 0.01         # R值最小合理范围
SPO2_R_MAX = 0.5          # R值最大合理范围

# --------- 1. 数据读取 ---------
print("[1/4] 正在读取数据...")
try:
    ppg = np.loadtxt("ch2.txt")    # 心率信号 (PPG)
    red = np.loadtxt("ch2.txt")    # 红光 (SpO2)
    infrared = np.loadtxt("ch3.txt")  # 红外光 (SpO2)
    assert len(ppg) == len(red) == len(infrared), "错误: ch2.txt 和 ch3.txt 长度不一致！"
    print(f"数据长度: {len(ppg)} (约 {len(ppg)/FS:.1f} 秒)")
except Exception as e:
    raise ValueError(f"数据读取失败: {e}")

# --------- 2. 心率提取 ---------
print("[2/4] 正在提取心率...")
def extract_hr(signal, fs):
    # 带通滤波
    nyq = fs / 2
    b, a = butter(2, [LOW_CUT/nyq, HIGH_CUT/nyq], btype='band')
    filtered = filtfilt(b, a, signal)
    
    # 峰值检测
    peaks, _ = find_peaks(filtered, distance=fs*0.6)  # 至少间隔0.6秒
    if len(peaks) < 2:
        raise ValueError("未检测到足够心率峰值！")
    
    # 计算连续心率
    rr_intervals = np.diff(peaks) / fs
    hr = 60 / rr_intervals
    hr_series = np.interp(np.arange(len(signal)), peaks[1:], hr)
    return hr_series

hr_series = extract_hr(ppg, FS)
print(f"心率范围: {np.nanmin(hr_series):.1f}-{np.nanmax(hr_series):.1f} BPM")

# --------- 3. 血氧提取 ---------
print("[3/4] 正在提取血氧...")
def extract_spo2(red, infrared, fs):
    spo2 = []
    for i in range(0, len(red), fs * STEP_SIZE):
        r_win = red[i:i + fs * WINDOW_SIZE]
        ir_win = infrared[i:i + fs * WINDOW_SIZE]
        
        # 检查信号有效性
        if len(r_win) == 0 or len(ir_win) == 0:
            spo2.append(98.0)  # 默认值
            continue
            
        # 计算AC/DC
        r_ac, r_dc = np.std(r_win), np.mean(r_win)
        ir_ac, ir_dc = np.std(ir_win), np.mean(ir_win)
        
        # 处理无效数据
        if r_dc <= 0 or ir_dc <= 0:
            spo2.append(98.0)  # 默认值
            continue
            
        # 计算R值并约束范围
        R = (r_ac / r_dc) / (ir_ac / ir_dc)
        R = np.clip(R, SPO2_R_MIN, SPO2_R_MAX)
        
        # 计算SpO2 (经验公式)
        spo2_val = -54.85 * R**2 + 32.4 * R + 100.6
        spo2.append(spo2_val)
    
    return np.array(spo2)

spo2_series = extract_spo2(red, infrared, FS)
print(f"血氧范围: {np.nanmin(spo2_series):.1f}-{np.nanmax(spo2_series):.1f}%")

# --------- 4. 对齐词语时间戳 ---------
print("[4/4] 正在对齐词语...")
try:
    word_df = pd.read_csv("4.csv")
    word_df["HR_Avg"] = 0.0  # 初始化列
    word_df["SpO2_Avg"] = 0.0
    
    for idx, row in word_df.iterrows():
        start = int(row["Start Time"] * FS)
        end = int(row["End Time"] * FS)
        
        # 强制索引不越界
        start = max(0, min(start, len(hr_series)-1))
        end = max(0, min(end, len(hr_series)-1))
        
        # 计算平均值 (确保无NaN)
        word_df.at[idx, "HR_Avg"] = np.mean(hr_series[start:end]) if start < end else hr_series[start]
        word_df.at[idx, "SpO2_Avg"] = np.mean(spo2_series[start//FS:end//FS]) if start < end else spo2_series[start//FS]

    # 保存结果
    word_df.to_csv("5.csv", index=False)
    print("处理完成！结果已保存。")
    
    # 检查缺失值
    missing_hr = word_df["HR_Avg"].isna().sum()
    missing_spo2 = word_df["SpO2_Avg"].isna().sum()
    print(f"缺失值统计: HR={missing_hr}, SpO2={missing_spo2}")
    
except Exception as e:
    raise RuntimeError(f"对齐失败: {e}")