## 项目概述

openWakeWord 是一个开源的唤醒词检测库，主要用于检测语音中的特定关键词（如"Alexa"、"Hey Jarvis"等）。

采用深度学习模型，支持实时音频流处理和离线模型训练。

## 核心架构

1. 主要模块结构

```
openwakeword/
├── __init__.py          # 包入口，模型路径配置
├── model.py            # 主模型类，唤醒词检测核心逻辑
├── vad.py              # 语音活动检测模块
├── utils.py            # 音频特征提取工具类
├── train.py            # 模型训练模块
├── data.py             # 数据处理模块
└── custom_verifier_model.py  # 自定义验证器模型

```

2. 核心流程解析
2.1 音频输入处理流程

```python
# 示例：从麦克风实时检测
import pyaudio
import numpy as np
from openwakeword.model import Model

# 初始化模型
owwModel = Model()

# 获取音频流
audio = pyaudio.PyAudio()
mic_stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True)

while True:
    # 读取音频数据
    audio_data = np.frombuffer(mic_stream.read(1280), dtype=np.int16)
    
    # 预测唤醒词
    prediction = owwModel.predict(audio_data)
    
    # 处理预测结果
    for wakeword, score in prediction.items():
        if score > 0.5:
            print(f"检测到唤醒词: {wakeword}, 置信度: {score}")

```

2.2 核心处理流程分解
1. 音频预处理(`AudioFeatures`类)
  - 16kHz,16-bit PCM音频输入
  - 可选 Speex 噪声抑制
  - 梅尔频谱图提取
  - Google Speech Embedding 特征提取
2. 语音活动检测(`VAD`类)
  - 基于Silero VAD 模型
  - 过滤非语音片段
  - 减少误报
3. 唤醒词检测(`Model`类)
  - 加载预训练模型(TFLite/ONNX)
  - 实时流势预测
  - 多模型并行处理

2.3 模型训练流程

```python
# 训练自定义唤醒词模型
from openwakeword.train import Model

# 1. 数据准备：生成正负样本
# 2. 特征提取：使用 AudioFeatures 提取特征
# 3. 模型训练：使用 PyTorch 训练 DNN 模型
# 4. 模型导出：转换为 ONNX/TFLite 格式

```

## 关键依赖关系

1.硬件/环境依赖
  - 音频输入:16kHz,16-bit PCM格式
  - 推理框架:TensorFlow Lite或ONNX Runtime
  - 可选依赖:SpeexDSP(噪声抑制)
2.软件依赖
```python
# 核心依赖
import numpy as np
import pyaudio  # 音频输入
import tflite_runtime  # 或 onnxruntime

# 训练依赖
import torch
import torchmetrics

```
3.模型文件依赖
项目预置了多个唤醒词模型

- `alexa_v0.1.tflite`
- `hey_jarvis_v0.1.tflite`
- `hey_mycroft_v0.1.tflite`
- 等...
