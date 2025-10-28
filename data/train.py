import openwakeword
from openwakeword.model import Model
import glob
import os

# 定义正样本和负样本路径
positive_paths = glob.glob("positive_samples/*.wav")
negative_paths = glob.glob("negative_samples/*.wav", recursive=True)

# 初始化 openWakeWord 模型
model = Model(
    wakeword_models=[],  # 自定义热词名称
    training_data={
        "positive": positive_paths,
        # "negative": negative_paths[:1000]  # 限制负样本数量以加快训练
    }
)

# 训练模型
model.train(
    output_path="data/hey_cc.tflite",
    epochs=10,  # 训练轮数
    batch_size=32
)
print("训练完成！模型保存至 models/hey_cc.tflite")
