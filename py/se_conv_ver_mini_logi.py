import os
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import tqdm
from torch.utils.data import Dataset, DataLoader
import time

import numpy as np


def conv_2d(in_planes, out_planes, stride=(1, 1), size=3):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, size), stride=stride,
                     padding=(0, (size - 1) // 2), bias=False)


def conv_1d(in_planes, out_planes, stride=1, size=3):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=size, stride=stride,
                     padding=(size - 1) // 2, bias=False)


class SEBlock1d(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        # return x * y.expand_as(x)
        return x * y


class SEBlock2d(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # return x * y.expand_as(x)
        return x * y


# Basic Blocks with SE
class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, size=3, res=True):
        super(BasicBlock1d, self).__init__()
        self.conv1 = conv_1d(inplanes, planes, stride, size=size)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_1d(planes, planes, size=size)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv_1d(planes, planes, size=size)
        self.bn3 = nn.BatchNorm1d(planes)
        self.se = SEBlock1d(planes)  # SE addition
        self.dropout = nn.Dropout(.2)
        self.downsample = downsample
        self.stride = stride
        self.res = res

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        # print(f'out:{out.shape}, x:{x.shape}')
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.se(out)  # SE application
        if self.res:
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
        return out


class BasicBlock2d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1, 1), downsample=None, size=3, res=True):
        super(BasicBlock2d, self).__init__()
        self.conv1 = conv_2d(inplanes, planes, stride, size=size)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_2d(planes, planes, size=size)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv_2d(planes, planes, size=size)
        self.bn3 = nn.BatchNorm2d(planes)
        self.se = SEBlock2d(planes)  # SE addition
        self.dropout = nn.Dropout(.2)
        self.downsample = downsample
        self.stride = stride
        self.res = res

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.se(out)  # SE application
        if self.res:
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
        return out


# ECGNet remains mostly the same except for SE additions above
class ECGNet(nn.Module):
    def __init__(self, input_channel=1, num_classes=2):
        # sizes = [
        #     [3, 3, 3, 3, 3, 3],
        #     [5, 5, 5, 5, 3, 3],
        #     [7, 7, 7, 7, 3, 3],
        # ]
        # layers = [
        #     [3, 3, 2, 2, 2, 2],
        #     [3, 2, 2, 2, 2, 2],
        #     [2, 2, 2, 2, 2, 2]
        # ]
        sizes = [
            [3, 3, 3, 3, 3],
            # [5, 5, 5, 5, 3],
            [7, 7, 7, 7, 3],
        ]
        layers = [
            [3, 3, 2, 2, 2],
            # [3, 2, 2, 2, 2],
            [2, 2, 2, 2, 2]
        ]

        self.sizes = sizes
        super(ECGNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=(1, 32), stride=(1, 2), padding=(0, 0),
                               bias=False)  # 50/32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(1, 8), stride=(1, 1), padding=(0, 0), bias=False)  # 16/8
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 0))
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.se = SEBlock2d(32)  # SE addition
        self.layers1_list = nn.ModuleList()
        self.layers2_list = nn.ModuleList()
        for i, size in enumerate(sizes):
            self.inplanes = 32
            self.layers1 = nn.Sequential()
            self.layers2 = nn.Sequential()
            self.layers1.add_module(f'layer{i}_1_1',
                                    self._make_layer2d(BasicBlock2d, 32, layers[i][0], stride=(1, 1), size=sizes[i][0]))
            self.layers1.add_module(f'layer{i}_1_2',
                                    self._make_layer2d(BasicBlock2d, 32, layers[i][1], stride=(1, 1), size=sizes[i][1]))
            self.inplanes *= 3
            self.layers2.add_module(f'layer{i}_2_1',
                                    self._make_layer1d(BasicBlock1d, 96, layers[i][2], stride=2, size=sizes[i][2]))
            self.layers2.add_module(f'layer{i}_2_2',
                                    self._make_layer1d(BasicBlock1d, 96, layers[i][3], stride=2, size=sizes[i][3]))
            # self.layers2.add_module(f'layer{i}_2_3',
            #                         self._make_layer1d(BasicBlock1d, 96, layers[i][4], stride=2, size=sizes[i][4]))
            # self.layers2.add_module(f'layer{i}_2_4',
            #                         self._make_layer1d(BasicBlock1d, 96, layers[i][5], stride=2, size=sizes[i][5]))
            self.layers1_list.append(self.layers1)
            self.layers2_list.append(self.layers2)

        # self.fc = nn.Linear(96 * len(sizes) + 2, num_classes)
        self.fc = nn.Linear(96 * len(sizes) + 2, 1)

    def _make_layer1d(self, block, planes, blocks, stride=2, size=3, res=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, padding=0, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, size=size, res=res))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, size=size, res=res))
        return nn.Sequential(*layers)

    def _make_layer2d(self, block, planes, blocks, stride=(1, 2), size=3, res=True):
        downsample = None
        if stride != (1, 1) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=(1, 1), padding=(0, 0), stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, size=size, res=res))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, size=size, res=res))
        return nn.Sequential(*layers)

    # def forward(self, x0):
    def forward(self, x0, fr):
        # print(x0.shape)
        x0 = x0.unsqueeze(1)
        # print(x0.shape, fr.shape)
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)
        x0 = self.conv2(x0)
        x0 = self.maxpool(x0)
        x0 = self.se(x0)
        # print(x0.shape)
        # return x0

        xs = []
        for i in range(len(self.sizes)):
            x = self.layers1_list[i](x0)
            x = torch.flatten(x, start_dim=1, end_dim=2)
            x = self.layers2_list[i](x)
            x = self.avgpool(x)
            xs.append(x)
        out = torch.cat(xs, dim=2)
        out = out.view(out.size(0), -1)

        # fr = torch.zeros(out.shape[0], 2).to('cuda:0')###
        out = torch.cat([out, fr], dim=1)
        # print(out.shape)
        out = self.fc(out)
        out = nn.Sigmoid()(out)
        # print(out.shape)
        # print('-----')
        return out


# Training Procedure
class ECGDataset(Dataset):
    def __init__(self, num_samples=1000, data=None, labels=None, fr=None, seq_length=512, num_features=2):

        if data is None or labels is None:
            self.num_samples = num_samples
            self.ecg_signals = torch.randn(num_samples, 3, seq_length)  # (num_samples, 8, seq_length)
            self.fr_features = torch.randn(num_samples, num_features)
            self.labels = torch.randint(0, 2, (num_samples,))
        else:
            self.num_samples = len(data)
            self.ecg_signals = torch.Tensor(data)
            if fr is None:
                self.fr_features = torch.zeros(self.num_samples, num_features)
            else:
                self.fr_features = torch.Tensor(fr)
                self.fr_features = torch.zeros(self.num_samples, num_features)
                assert self.fr_features.shape == torch.zeros(self.num_samples, num_features).shape
            # self.labels = torch.LongTensor(labels)
            self.labels = torch.Tensor(labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (self.ecg_signals[idx], self.fr_features[idx]), self.labels[idx]


import datetime
import onnx

if __name__ == '__main__':
    save_date = datetime.datetime.now().strftime("%m%d%H")
    '''
    model = ECGNet()
    model.load_state_dict(torch.load("models/best_modelm_060610_7_9981.pth"))
    ecg = torch.randn(1, 3, 256)
    fr = torch.randn(1, 2)
    torch.onnx.export(model, (ecg,fr), "ecg_net_model.onnx",
                  input_names=['signal', 'enhance'],
                  output_names=['output'],
                  export_params=True,
                  opset_version=13, do_constant_folding=True,
                  verbose=True)

    onnx_model = onnx.load("ecg_net_model.onnx")
    onnx.checker.check_model(onnx_model)
    print("多输入模型已成功导出并通过验证。")

for input_tensor in onnx_model.graph.input:
    print('Input Name:', input_tensor.name)
    shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
    print('Input Shape:', shape)

for output_tensor in onnx_model.graph.output:
    print('Output Name:', output_tensor.name)
    shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
    print('Output Shape:', shape)

    #out = model(ecg, fr)
    timeS = time.time()
    #for i in range(1000):
    #    out = model(ecg, fr)
    timeE = time.time()
    print("Time:", timeE - timeS)
    #print("Output shape:", out.shape)

    '''
    model = ECGNet(1, 2)
    model.eval()
    ecg = torch.randn(1, 3, 256)
    fr = torch.randn(1, 2)
