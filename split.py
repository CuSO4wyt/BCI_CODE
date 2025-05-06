import whisper
import pandas as pd

# 加载 Whisper 模型
model = whisper.load_model("base")

# 识别音频并提取每个词语及其时间戳
def transcribe_audio_whisper(audio_file):
    result = model.transcribe(audio_file, word_timestamps=True)  # 启用词语时间戳
    word_timestamps = []

    # 遍历每个词的时间戳
    for segment in result["segments"]:
        for word_info in segment["words"]:
            word = word_info["word"]
            start_time = word_info["start"]
            end_time = word_info["end"]
            word_timestamps.append((word, start_time, end_time, 1))  # 默认Label为1（真）

    return word_timestamps

# 保存数据到 CSV 文件
def save_to_csv(word_timestamps, output_file):
    # 将数据转化为 DataFrame 并加上默认的 Label 列
    df = pd.DataFrame(word_timestamps, columns=["Word", "Start Time", "End Time", "Label"])
    df.to_csv(output_file, index=False)
    print(f"数据已保存到 {output_file}")

# 主程序
if __name__ == "__main__":
    audio_file = 'converted.wav'  # 输入的音频文件路径
    output_file = 'word_timestamps.csv'  # 输出的 CSV 文件路径

    # 提取音频中的词和时间戳
    word_timestamps = transcribe_audio_whisper(audio_file)

    # 打印提取的一部分数据（可选）
    if word_timestamps:
        for word, start_time, end_time, label in word_timestamps[:5]:
            print(f"Word: {word}, Start Time: {start_time}, End Time: {end_time}, Label: {label}")

        # 保存数据到 CSV 文件（包括空的 Label 列用于后续手动标注真假）
        save_to_csv(word_timestamps, output_file)
    else:
        print("没有提取到任何数据，请检查音频文件内容。")
