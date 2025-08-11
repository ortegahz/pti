import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper

onnx_path = "/home/manu/tmp/ecg_net_model_sim.onnx"
layer_blob = "/Unsqueeze_output_0"  # 你想观察的 blob 名

# ------------------------------------------------------------------
# 1. 让 conv1 输出成为 graph 输出
# ------------------------------------------------------------------
model = onnx.load(onnx_path)

# 如果已经在输出列表里就不必重复添加
if layer_blob not in {o.name for o in model.graph.output}:
    vi = helper.ValueInfoProto()
    vi.name = layer_blob
    model.graph.output.append(vi)  # 追加
    dump_path = "/home/manu/tmp/ecg_with_conv1.onnx"
    onnx.save(model, dump_path)
else:
    dump_path = onnx_path  # 已有则直接用

# ------------------------------------------------------------------
# 2. 创建 session
# ------------------------------------------------------------------
sess = ort.InferenceSession(dump_path, providers=["CPUExecutionProvider"])

# ------------------------------------------------------------------
# 3. 组装输入，与之前一样
# ------------------------------------------------------------------
txt_path = "/home/manu/tmp/test_0.txt"
vec = np.loadtxt(txt_path, dtype=np.float32)
signal = vec[:768].reshape(1, 3, 256).astype(np.float32)
enhance = vec[768:770].reshape(1, 2).astype(np.float32)

inputs = {"signal": signal, "enhance": enhance}

# print(signal[0].reshape(3, 256)[:, :10])  # 去掉 batch 维后与 ncnn 的 (3,256) 对齐

# ------------------------------------------------------------------
# 4. 推理：显式指定要拿到的输出名
# ------------------------------------------------------------------
output_names = ["output", layer_blob]  # “output” 是你真正的网络输出
out_main, feat = sess.run(output_names, inputs)

print("main output =", out_main.flatten()[:10])
print("conv1 feat  =", feat.flatten()[:10])
