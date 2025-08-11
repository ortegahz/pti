import ncnn

param_path = '/home/manu/tmp/ecg_net_model.param'
model_path = '/home/manu/tmp/ecg_net_model.bin'
txt_path = '/home/manu/tmp/test_0.txt'

import numpy as np

w = np.fromfile(model_path, dtype=np.float32)
print(np.isnan(w).any(), np.isinf(w).any())  # True 就说明权重里本身就有 NaN/Inf

vec = np.loadtxt(txt_path, dtype=np.float32)
signal_raw, enhance_raw, label = vec[:768], vec[768:770], int(vec[770])

# ---------- 关键：按 (c,h,w) ----------
sig = signal_raw.reshape(3, 1, 256)  # (3,1,256)
signal_mat = ncnn.Mat(sig)  # dims = 3 → w=256,h=1,c=3
enhance_mat = ncnn.Mat(enhance_raw)  # dims = 1

print('signal_mat :', signal_mat.w, signal_mat.h, signal_mat.c)  # 256 1 3

net = ncnn.Net()
net.load_param(param_path)
net.load_model(model_path)

ex = net.create_extractor()
ex.input('signal', signal_mat)  # 名称对齐 .param
ex.input('enhance', enhance_mat)

# ret, in_blob = ex.extract('signal')  # ncnn 的原始输入 blob
# print(np.array(in_blob).reshape(3, 256)[:, :10])

ret, feat = ex.extract('/Unsqueeze_output_0')
print(np.array(feat).ravel()[:10])

ret, out = ex.extract('output')
if ret != 0:
    raise RuntimeError(f'extract failed, code = {ret}')

logit = float(np.array(out).flatten()[0])
prob = 1 / (1 + np.exp(-logit))

print('logit ->', logit)
print('prob  ->', prob, ' label =', label)
print('pred  ->', int(prob > 0.5))
