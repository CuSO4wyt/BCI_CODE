import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.signal import butter, filtfilt, find_peaks

# ---------- Step 1: 读取数据 ----------
fs_hr = 200  # 心率采样率
fs_spo2 = 200  # 血氧采样率

ppg_signal = np.loadtxt("C:\\Users\\SIRIU\\Desktop\\3.9-red1.txt")
labels = np.loadtxt("C:\\Users\\SIRIU\\Desktop\\output_labels.txt").astype(int)
red = np.loadtxt("C:\\Users\\SIRIU\\Desktop\\3.9-red1.txt")         # 660nm
infrared = np.loadtxt("C:\\Users\\SIRIU\\Desktop\\3.9-red-2.txt")  # 880nm

# ---------- Step 2: 提取心率 ----------
def bandpass_filter(signal, fs, low=0.5, high=4.0): 
    '''
    signal:ppg 输入的PPG原始信号
    fs: 采样率
    low 和 high: 带通滤波的低/高截止频率
    '''
    nyq = fs / 2
    b, a = butter(2, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)

def extract_hr(signal, fs, window_size=5, step_size=1):
    '''
    window_size:窗口长度,即每次用5秒数据来估计一次心率。

    step_size: 每次窗口向前移动的步长(秒),默认为1秒。

    total_points: 信号的总采样点数。

    hr_series: 存储每个窗口的心率。

    该函数用于按窗口提取,计算心率数据,并存储到hr_series
    '''
    signal = bandpass_filter(signal, fs)
    total_points = len(signal)
    #print(f"Total points in signal: {total_points}")  # Debug: Check signal length
    hr_series = []
    for start in range(0, total_points - window_size * fs, step_size * fs):
        window = signal[start:start + window_size * fs]
        #print(f"Processing window from {start} to {start + window_size * fs}")  # Debug: Check window range
        peaks, _ = find_peaks(window, distance=fs*0.4)
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) / fs
            hr = np.mean(60 / rr_intervals)
        else:
            hr = 0
        hr_series.append(hr)
        #print(f"Heart rate for this window: {hr}")  # Debug: Print the heart rate
    return np.array(hr_series)

hr_series = extract_hr(ppg_signal, fs=fs_hr)  # 200Hz

# ---------- Step 3: 提取血氧 ----------
def extract_spo2(red, infrared, fs, window_size=5, step_size=1):
    """
    从红光、红外信号中提取 SpO2 时间序列,存储到spo2_seires中
    """
    spo2_series = []
    for start in range(0, len(red) - window_size*fs, step_size*fs):
        r_win = red[start:start + window_size * fs]
        ir_win = infrared[start:start + window_size * fs]
        # AC 为标准差，DC 为均值
        r_ac, r_dc = np.std(r_win), np.mean(r_win)
        ir_ac, ir_dc = np.std(ir_win), np.mean(ir_win)
        if r_dc == 0 or ir_dc == 0:
            spo2 = 0
        else:
            R = (r_ac / r_dc) / (ir_ac / ir_dc)
            spo2 = -54.85 * R**2 + 32.4 * R + 100.6
        spo2_series.append(spo2)
    return np.array(spo2_series)

spo2_series = extract_spo2(red, infrared, fs=fs_spo2)
#print("len(ppg_signal):", len(ppg_signal))
#print("len(hr_series):", len(hr_series))
#print("len(spo2_series):", len(spo2_series))
#print("len(labels):", len(labels))


# ---------- Step 4: 对齐心率和血氧 ----------
# SpO2是200Hz下提取的，每秒一个点，所以长度差不多，先统一为最短长度


min_len = min(len(hr_series), len(spo2_series), len(labels))
hr_series = hr_series[:min_len]
spo2_series = spo2_series[:min_len]
labels = labels[:min_len]  # 不进行下采样，直接裁剪

#debug
#print("len(hr_series):", len(hr_series))
#print("len(spo2_series):", len(spo2_series))
#print("len(labels):", len(labels) )




# 合并为多通道特征
X = np.stack([hr_series, spo2_series], axis=1)  # [T, 2] 心率+血氧
#print("X shape:", X.shape)  # 应该是 (T, 2)
y = labels

# ---------- Step 5: 构建序列 ----------

#这一部分每10s提取一次特征，并且将X和y数据切换为NumPy数组
sequence_length = 2  # 10秒窗口
X_seq, y_seq = [], []

for i in range(len(X) - sequence_length):
    X_seq.append(X[i:i+sequence_length])  # [10, 2]
    label_window = y[i:i+sequence_length]
    y_seq.append(1 if np.any(label_window == 1) else 0)


X_seq = np.array(X_seq)       # [N, 10, 2]
y_seq = np.array(y_seq)       # [N]

#print("X_seq shape:", X_seq.shape)  # 应该是 (N, 10, 2)

# ---------- Step 6: 构建 DataLoader ----------

#将数据转换为 PyTorch 张量
X_tensor = torch.tensor(X_seq, dtype=torch.float32)  # 不需要 unsqueeze
y_tensor = torch.tensor(y_seq, dtype=torch.long)

#划分训练集和测试集
train_X, test_X, train_y, test_y = train_test_split(X_tensor, y_tensor, test_size=0.1, random_state=42)
train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=32)

# ---------- Step 7: 定义 LSTM 模型 ----------
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=32, batch_first=True)  # 改为2通道
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 获取序列的最后一个时间步的输出
        return self.fc(out)  # 将LSTM的输出通过全连接层

model = LSTMClassifier() # 创建模型对象
loss_fn = nn.CrossEntropyLoss() # 用于多类分类问题的交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # 使用 Adam 优化器来更新模型的参数

# ---------- Step 8: 训练 ----------
for epoch in range(10):
    model.train()
    for batch_X, batch_y in train_loader:
        #batch_X, batch_y = batch_X.to(device), batch_y.to(device)  # 确保数据和模型在相同设备上

        pred = model(batch_X)  # 前向传播
        loss = loss_fn(pred, batch_y)  # 计算损失
        optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

    #print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

# ---------- Step 9: 测试 ----------
model.eval()  # 切换到评估模式
all_preds, all_labels = [], []
with torch.no_grad():  # 禁用梯度计算，节省内存
    for batch_X, batch_y in test_loader:
        #batch_X, batch_y = batch_X.to(device), batch_y.to(device)  # 确保数据和模型在相同设备上
        
        pred = model(batch_X)  # 前向传播
        pred_class = torch.argmax(pred, dim=1)  # 选取概率最大的类别
        print(f"Predictions: {pred_class.cpu().numpy()} | Actual Labels: {batch_y.cpu().numpy()}")  # 打印预测值和实际标签
        all_preds.extend(pred_class.cpu().numpy())  # 移动数据到CPU并转换为numpy数组
        all_labels.extend(batch_y.cpu().numpy())  # 同上，标签也要转到CPU

# 计算测试集准确率
acc = accuracy_score(all_labels, all_preds)  # 使用sklearn计算准确率
print(f"Test size: 0.1")
print(f"✅ Test Accuracy: {acc * 100:.2f}%")

