import numpy as np
import onnxruntime as ort


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------
# 1. 载入 ONNX 模型
# ---------------------------------------------------------------
onnx_path = "/home/manu/tmp/ecg_net_model_sim.onnx"
sess = ort.InferenceSession(
    onnx_path,
    providers=["CPUExecutionProvider"]  # 如果想用 GPU 可改为 ["CUDAExecutionProvider"]
)

# ---------------------------------------------------------------
# 2. 读取同一份保存的 txt，并拆分成 signal / enhance / label
# ---------------------------------------------------------------
txt_path = "/home/manu/tmp/test_0.txt"
vec = np.loadtxt(txt_path, dtype=np.float32)

signal = vec[:768].reshape(1, 3, 256).astype(np.float32)  # (1,3,256)
enhance = vec[768:770].reshape(1, 2).astype(np.float32)  # (1,2)
label = int(vec[770])  # 最后一个作为标签

# ---------------------------------------------------------------
# 3. ONNX Runtime 推理
# ---------------------------------------------------------------
inputs = {
    "signal": signal,
    "enhance": enhance
}
outputs = sess.run(None, inputs)  # 输出是一个 list（因为只有1个 output）
logit = outputs[0][0]  # 假设输出 shape=(1,) 取标量
prob = sigmoid(logit)

print(f"logit --> {logit}")
