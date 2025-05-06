import whisper

# 加载 Whisper 模型
model = whisper.load_model("base")

# 识别音频
def transcribe_audio_whisper(audio_file):
    result = model.transcribe(audio_file)
    return result["text"]

# 使用 Whisper 识别中文音频
audio_file = 'converted.wav'  # 替换为你自己的音频文件路径
transcript = transcribe_audio_whisper(audio_file)

print("识别的文本：", transcript)
