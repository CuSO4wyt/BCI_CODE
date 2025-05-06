import re
from docx import Document

def extract_time_and_labels(doc_path):
    doc = Document(doc_path)
    timestamps = []
    labels_by_time = {}

    for para in doc.paragraphs:
        # 提取该段落所有时间戳（格式如 01:12）
        time_matches = re.findall(r'\b\d{1,2}:\d{2}\b', para.text)
        if not time_matches:
            continue
        
        for time_match in time_matches:
            # 计算该时间戳对应的总秒数
            mins, secs = map(int, time_match.split(":"))
            total_seconds = mins * 60 + secs
            timestamps.append(total_seconds)

            # 找到该时间戳后面的内容是否有高亮
            is_highlighted = False
            found_time = False

            for run in para.runs:
                if not found_time and time_match in run.text:
                    found_time = True
                    idx = run.text.find(time_match) + len(time_match)
                    if idx < len(run.text) and run.font.highlight_color:
                        is_highlighted = True
                elif found_time:
                    if run.font.highlight_color:
                        is_highlighted = True
            
            labels_by_time[total_seconds] = 1 if is_highlighted else 0

    return sorted(timestamps), labels_by_time

def generate_labels(timestamps, labels_by_time, end_time):
    result = []
    for i in range(len(timestamps)):
        start = timestamps[i]
        end = timestamps[i + 1] if i + 1 < len(timestamps) else end_time
        label = labels_by_time[start]
        duration = end - start
        result.extend([label] * duration)
    return result

def main():
    doc_path = "C:\\Users\\SIRIU\\Desktop\\3.9-test1.docx"  # 确保脚本和文件在同一目录，或写入完整路径
    timestamps, labels_by_time = extract_time_and_labels(doc_path)
    
    # 如果你知道视频总时长，可以修改这里
    estimated_end_time = 4 * 60  # 假设为5分钟

    labels = generate_labels(timestamps, labels_by_time, end_time=estimated_end_time)

    # 输出到 txt 文件
    with open("C:\\Users\\SIRIU\\Desktop\\output_labels.txt", "w") as f:
        f.write("\n".join(str(label) for label in labels))

    print(f"输出成功，共生成 {len(labels)} 个标签。保存在 output_labels.txt。")

if __name__ == "__main__":
    main()
