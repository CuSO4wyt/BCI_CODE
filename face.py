import cv2
import dlib
import pandas as pd
import numpy as np

# 加载 dlib 的面部检测器和面部关键点预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 下载 dlib 面部关键点预测模型

# 提取面部特征（眼睛、嘴巴、眉毛、下巴等关键点）
def extract_face_features(video_file, start_time, end_time, frame_rate=30):
    # 打开视频文件
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
    frame_start = int(start_time * fps)
    frame_end = int(end_time * fps)
    
    # 跳到开始的帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    
    # 读取视频帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    ret, frame = cap.read()
    
    if not ret:
        return None

    # 转为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测面部
    faces = detector(gray)
    face_features = []

    for face in faces:
        landmarks = predictor(gray, face)  # 获取面部关键点

        # 获取眼睛、嘴巴、眉毛、下巴等关键点坐标
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)
        mouth = (landmarks.part(48).x, landmarks.part(48).y)
        left_eyebrow = (landmarks.part(22).x, landmarks.part(22).y)
        right_eyebrow = (landmarks.part(26).x, landmarks.part(26).y)
        chin = (landmarks.part(8).x, landmarks.part(8).y)

        # 计算眼睛、嘴巴、眉毛的中心坐标
        eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        mouth_center = mouth
        eyebrow_center = ((left_eyebrow[0] + right_eyebrow[0]) // 2, (left_eyebrow[1] + right_eyebrow[1]) // 2)
        chin_center = chin

        # 计算面部表情强度：微笑、皱眉、眼睛闭合等
        smile_intensity = abs(mouth[0] - landmarks.part(54).x)  # 使用嘴巴左右距离来衡量微笑强度
        frown_intensity = abs(left_eyebrow[1] - right_eyebrow[1])  # 眉毛间距来衡量皱眉强度
        eye_closure = abs(left_eye[1] - right_eye[1])  # 眼睛的闭合强度

        # 计算面部动作单位（AU）
        AU6 = abs(left_eye[1] - right_eye[1])  # 眼睛眯起（估算）
        AU12 = smile_intensity  # 微笑（嘴角上扬）
        AU1_2 = abs(left_eyebrow[1] - right_eyebrow[1])  # 眉毛升降

        # 存储面部特征
        face_features.append([eye_center, mouth_center, eyebrow_center, chin_center, smile_intensity, frown_intensity, eye_closure, AU6, AU12, AU1_2])

    # 释放视频文件
    cap.release()
    
    return face_features

# 读取 CSV 文件
input_file = '2.csv'  # 输入的 CSV 文件路径
df = pd.read_csv(input_file)

# 假设视频文件是 'example_video.mp4'
video_file = 'zyl.mp4'

# 为每一行添加面部特征列
face_features_list = []

for _, row in df.iterrows():
    start_time = row['Start Time']
    end_time = row['End Time']
    
    # 提取当前词对应的面部特征
    face_features = extract_face_features(video_file, start_time, end_time)
    
    # 如果检测到面部特征，则提取面部的相关信息
    if face_features:
        (eye_center, mouth_center, eyebrow_center, chin_center, 
         smile_intensity, frown_intensity, eye_closure, AU6, AU12, AU1_2) = face_features[0]
        
        face_features_list.append([eye_center[0], eye_center[1], mouth_center[0], mouth_center[1],
                                   eyebrow_center[0], eyebrow_center[1], chin_center[0], chin_center[1],
                                   smile_intensity, frown_intensity, eye_closure, AU6, AU12, AU1_2])
    else:
        face_features_list.append([None] * 14)  # 填充 None，如果没有面部检测到

# 将面部特征列添加到 DataFrame 中
face_features_df = pd.DataFrame(face_features_list, columns=[
    "Left Eye X", "Left Eye Y", "Mouth X", "Mouth Y", "Eyebrow X", "Eyebrow Y", "Chin X", "Chin Y",
    "Smile Intensity", "Frown Intensity", "Eye Closure", "AU6", "AU12", "AU1_2"
])

df = pd.concat([df, face_features_df], axis=1)

# 保存更新后的 CSV 文件
output_file = '3.csv'  # 输出文件路径
df.to_csv(output_file, index=False)

print(f"数据已保存到 {output_file}")
