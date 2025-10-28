# 使用 gTTS 快速生成大量训练数据
from gtts import gTTS
from pydub import AudioSegment
import os

# 目标热词
wake_word = "hey cc"

# 创建输出目录
os.makedirs("positive_samples", exist_ok=True)

# 生成 10 个样本（可根据需要增加）
for i in range(10):
    # 使用 gTTS 生产 MP3
    mp3_path = f"positive_samples/hey_cc_{i}.mp3"
    tts = gTTS(text=wake_word, lang='zh-cn')
    tts.save(mp3_path)

    # 使用 AudioSegment 转换为 wav
    wav_path = f"positive_samples/hey_cc_{i}.wav"
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")

    # 删除临时生成的 MP3 文件(可选)
    os.remove(mp3_path)
    print(f"生成样本 {i+1}/100")
